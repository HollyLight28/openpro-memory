import { describe, test, expect, vi, beforeEach } from "vitest";
import { handleRecall, handleCapture } from "./handlers.js";
import { MemoryDB } from "./database.js";
import { Embeddings } from "./embeddings.js";
import { ChatModel } from "./chat.js";
import { GraphDB } from "./graph.js";
import { DreamService } from "./dream.js";
import { WorkingMemoryBuffer } from "./buffer.js";
import { ConversationStack } from "./stack.js";
import { MemoryTracer } from "./tracer.js";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { rm } from "node:fs/promises";

describe("Memory Hybrid E2E Integration", () => {
  let db: MemoryDB;
  let embeddings: Embeddings;
  let chatModel: ChatModel;
  let graphDB: GraphDB;
  let dreamService: DreamService;
  let workingMemory: WorkingMemoryBuffer;
  let conversationStack: ConversationStack;
  let tracer: MemoryTracer;
  let deps: any;
  const dbPath = join(tmpdir(), `memory-e2e-${Date.now()}`);

  beforeEach(async () => {
    try { await rm(dbPath, { recursive: true, force: true }); } catch (e) {}
    
    tracer = new MemoryTracer({ customPath: join(dbPath, "traces.jsonl") });
    db = new MemoryDB(dbPath, 1536, tracer);
    embeddings = new Embeddings("key", "text-embedding-3-small");
    chatModel = new ChatModel("key", "gpt-3.5-turbo", "openai", tracer);
    graphDB = new GraphDB(dbPath, tracer);
    workingMemory = new WorkingMemoryBuffer(10, 0.5, 1); 
    conversationStack = new ConversationStack(10);
    dreamService = new DreamService({} as any, db, embeddings, graphDB, chatModel, tracer);

    deps = {
      db,
      embeddings,
      chatModel,
      graphDB,
      workingMemory,
      conversationStack,
      dreamService,
      tracer,
      cfg: {
        autoRecall: true,
        autoCapture: true,
        smartCapture: false, 
        captureMaxChars: 1000,
        dbPath: dbPath,
        embedding: { apiKey: "key", model: "text-embedding-3-small" }
      }
    };

    vi.spyOn(embeddings, "embed").mockResolvedValue(new Array(1536).fill(0.1));
    vi.spyOn(chatModel, "complete").mockResolvedValue("Summary: Test memory content");
  });

  test("Full Lifecycle: Capture -> Promotion -> Storage -> Recall", async () => {
    const api = {
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
      on: vi.fn(),
      registerService: vi.fn(),
      resolvePath: (p: string) => join(dbPath, p)
    } as any;

    const event1 = {
      success: true,
      messages: [
        { role: "user", content: "Мій улюблений колір — синій." },
        { role: "assistant", content: "Зрозумів, синій — твій улюблений колір!" }
      ]
    };
    
    await handleCapture(event1, { trigger: "user" }, api, deps, tracer);
    expect(workingMemory.size).toBe(1);

    workingMemory.add("Мій улюблений колір — синій.", 0.9, "preference");
    await handleCapture(event1, { trigger: "user" }, api, deps, tracer);

    const event2 = {
      prompt: "Який мій улюблений колір?"
    };

    const recallResult = await handleRecall(event2, { trigger: "user" }, api, deps, tracer);
    expect(recallResult).toBeDefined();
    expect(recallResult).toContain("синій");
  });
});
