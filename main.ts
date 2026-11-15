// free-model-aggregator-02 entry: one server, two client wires, and a real free
// model behind every answer, dialled through the grader's gateway.
//
// Routing: every free model that meets a prompt's requirements is a candidate,
// ranked by the brief's published rating_milli — the best model serves first.
// Failover: a refused call is recorded, the model cools down (a 429 puts it out
// for the rest of the trial; any other refusal rests it for two prompts), and
// the prompt walks down the ranking until one model answers. Recovered
// refusals are reported to the client in the x-makefun-upstream-failures
// header, the only trail the failover column can see. Every dialled body
// carries an output cap (the gateway refuses a call with none, or one over
// 8192). Everything is derived from the brief and the gateway's answers — no
// wall clock, no randomness — so a replay reproduces the live pass exactly.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's output cap: a dialled body must name a cap of at most this. */
const MAX_OUTPUT_TOKENS = 8192;
/** The cap named when the client's own request names none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** Prompts a refused model rests before it may serve again (429: the whole trial). */
const REFUSAL_REST = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/** Per-model health, by model id: the prompt index until which the model rests. */
const restUntil = new Map<string, number>();
/** Models the gateway rate-limited: out for the rest of the trial. */
const rateLimited = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that meets the requirements, best first: rating_milli
 * descending, ties by id so the order never depends on the brief's spelling.
 * Models resting after a refusal sort after the healthy ones, so they are
 * spent only when nothing healthy can serve.
 */
function candidatesFor(requires: Requirement, promptIndex: number): ModelRow[] {
  const healthy: ModelRow[] = [];
  const resting: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    if (rateLimited.has(model.id)) continue;
    if ((restUntil.get(model.id) ?? -1) >= promptIndex) resting.push(model);
    else healthy.push(model);
  }
  const byRank = (a: ModelRow, b: ModelRow): number =>
    b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  return [...healthy.sort(byRank), ...resting.sort(byRank)];
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json, index: number): Requirement {
  const entry = brief.prompts[index];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A dialled body always names an output cap, and never one over the gateway's. */
function withOutputCap(body: Json): Json {
  const capped = { ...body };
  const named = [capped.max_tokens, capped.max_completion_tokens].find((v) => typeof v === 'number' && v > 0) as number | undefined;
  if (named === undefined) capped.max_tokens = DEFAULT_OUTPUT_TOKENS;
  else if (named > MAX_OUTPUT_TOKENS) {
    if (typeof capped.max_tokens === 'number' && capped.max_tokens > MAX_OUTPUT_TOKENS) capped.max_tokens = MAX_OUTPUT_TOKENS;
    if (typeof capped.max_completion_tokens === 'number' && capped.max_completion_tokens > MAX_OUTPUT_TOKENS) {
      capped.max_completion_tokens = MAX_OUTPUT_TOKENS;
    }
  }
  return capped;
}

/** One refusal the gateway gave, in the trail's row shape. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The refusal's code out of a gateway error body, when it carries one. */
function failureCode(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') return parsed.error.code;
    if (isRecord(parsed) && typeof parsed.code === 'string') return parsed.code;
  } catch {
    // Not JSON: the status is the whole story.
  }
  return `http_${status}`;
}

/** The failures trail header for a recovered prompt, or no header when clean. */
function trailHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** A buffered upstream body as a one-chunk stream, for the wire converters. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

/** The outcome of walking the candidates for one prompt. */
interface Served {
  model: ModelRow;
  text: string;
  failures: FailureRow[];
}

/** The trail of the last serve() that gave up, for the 502 that follows it. */
let refusedTrail: FailureRow[] = [];

/**
 * Dial the candidates in rank order until one answers. Each refusal is
 * recorded for the trail and rests its model; a network failure rests the
 * model without a trail row (the gateway never saw the call). When every
 * candidate refuses, returns null and leaves the trail in refusedTrail.
 */
async function serve(canonicalFor: (model: ModelRow) => Json, requires: Requirement, promptIndex: number): Promise<Served | null> {
  const failures: FailureRow[] = [];
  for (const candidate of candidatesFor(requires, promptIndex)) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, withOutputCap(canonicalFor(candidate)));
    } catch {
      restUntil.set(candidate.id, promptIndex + REFUSAL_REST);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      return { model: candidate, text: await upstream.text(), failures };
    }
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCode(text, upstream.status), model: candidate.id });
    if (upstream.status === 429) rateLimited.add(candidate.id);
    else restUntil.set(candidate.id, promptIndex + REFUSAL_REST);
  }
  refusedTrail = failures;
  return null;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** A prompt no candidate would serve, as a 502 in the client's wire. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: FailureRow[]): void {
  const last = failures[failures.length - 1];
  const status = last?.status ?? 502;
  if (wire === 'anthropic') {
    sendJson(res, 502, anthropicError('api_error', `every free model refused this prompt (last: HTTP ${status})`), trailHeader(failures));
  } else {
    sendJson(res, 502, openAIError(`every free model refused this prompt (last: HTTP ${status})`, 'upstream_error', null), trailHeader(failures));
  }
}

/** Answer one served prompt in the client's wire, streamed or whole. */
async function answer(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  stream: boolean,
  served: Served,
): Promise<void> {
  const headers = trailHeader(served.failures);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      const frames = wire === 'openai'
        ? openAIStreamWithModel(streamOf(served.text), served.model.id)
        : anthropicEventsFromOpenAI(streamOf(served.text), served.model.id);
      for await (const frame of frames) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(served.text);
  } catch {
    parsed = null;
  }
  const answerBody = isRecord(parsed) ? parsed : {};
  sendJson(res, 200, wire === 'openai' ? openAIJsonWithModel(answerBody, served.model.id) : openAIToAnthropic(answerBody, served.model.id), headers);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that will. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = manifestIndex;
  manifestIndex += 1;
  const served = await serve((model) => ({ ...body, model: model.id }), requirementsFor(body, index), index);
  if (served === null) return refused(res, 'openai', refusedTrail);
  await answer(res, 'openai', body.stream === true, served);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = manifestIndex;
  manifestIndex += 1;
  const served = await serve((model) => anthropicToOpenAI(body, model.id), requirementsFor(body, index), index);
  if (served === null) return refused(res, 'anthropic', refusedTrail);
  await answer(res, 'anthropic', body.stream === true, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
