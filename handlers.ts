import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { HookDeps } from "./hooks.js";
import { hybridScore, getGraphEnrichment } from "./recall.js";
import { formatRadarContext, smartCapture, shouldCapture, detectCategory, generateMemorySummary } from "./capture.js";
import { TaskPriority } from "./limiter.js";
import { validateMemoryInput } from "./security.js";
import { extractGraphFromBatch } from "./graph.js";
import type { MemoryCategory } from "./config.js";
import { MemoryTracer, type Logger } from "./tracer.js";
import type { MemoryDB } from "./database.js";
import type { Embeddings } from "./embeddings.js";
import type { GraphDB } from "./graph.js";
import type { ChatModel } from "./chat.js";

/**
 * Handle memory recall (before_agent_start)
 */
export async function handleRecall(
  event: { prompt: string },
  ctx: any,
  api: OpenClawPluginApi,
  deps: HookDeps,
  tracer: MemoryTracer,
) {
  const { db, embeddings, graphDB, cfg } = deps;
  if (!event.prompt || event.prompt.length < 5) return;

  const nPrompt = event.prompt.trim().toLowerCase();
  // Skip commands and greetings
  if (nPrompt.startsWith("/") || /^(hi|hello|hey|привіт|вітаю)/i.test(nPrompt)) return;

  try {
    const isDeepTopic = /trauma|childhood|fear|secret|life|history/i.test(nPrompt);
    const limit = isDeepTopic ? 30 : 5;

    const vector = await embeddings.embed(event.prompt);
    const rawResults = await db.searchWithAMHR(vector, limit, graphDB, 0.3);
    const scored = await hybridScore(rawResults, graphDB);
    const finalScored = scored.slice(0, limit);

    const radarContext = formatRadarContext(
      finalScored.map((r) => ({
        id: r.entry.id,
        category: r.entry.category as MemoryCategory,
        summary: r.entry.summary,
        text: r.entry.text,
      })),
    );

    const graphInfo = await getGraphEnrichment(finalScored, graphDB);
    let context = radarContext;
    if (graphInfo) {
      context = context.replace("</star-map>", graphInfo + "\n</star-map>");
    }

    db.incrementRecallCount(finalScored.map((r) => r.entry.id));
    tracer.traceRecall(
      event.prompt,
      finalScored.map((s) => ({
        id: s.entry.id,
        text: s.entry.text,
        score: s.finalScore,
      })),
    );

    return { prependContext: context };
  } catch (err) {
    api.logger.warn(`memory-hybrid: recall failed: ${err}`);
  }
}

/**
 * Handle memory capture (agent_end)
 */
export async function handleCapture(
  event: any,
  ctx: any,
  api: OpenClawPluginApi,
  deps: HookDeps,
  tracer: MemoryTracer,
) {
  const { db, embeddings, chatModel, graphDB, conversationStack, workingMemory, cfg } = deps;
  const logger = api.logger;

  // Extract messages
  const userTexts: string[] = [];
  const assistantTexts: string[] = [];
  for (const msg of (event.messages || [])) {
    if (msg.role === "user") userTexts.push(msg.content);
    else if (msg.role === "assistant") assistantTexts.push(msg.content);
  }

  const lastUserMsg = userTexts.at(-1) || "";
  const lastAssistantMsg = assistantTexts.at(-1) || "";

  // 1. Conversation Stack (Context Compression)
  if (lastUserMsg.length > 10 || lastAssistantMsg.length > 10) {
    await conversationStack.push(lastUserMsg, lastAssistantMsg, chatModel, tracer, logger).catch(e => 
      logger.warn(`stack compression failed: ${e}`)
    );
  }

  // 2. Smart Capture (LLM-based)
  if (cfg.smartCapture && lastUserMsg.length > 15) {
    const validation = validateMemoryInput(lastUserMsg, cfg.captureMaxChars);
    if (validation.isValid) {
      const result = await smartCapture(lastUserMsg, lastAssistantMsg, chatModel, tracer, logger);
      if (result.shouldStore && result.facts.length > 0) {
        await processFacts(result.facts.slice(0, 5), db, embeddings, graphDB, chatModel, tracer, logger);
      }
    }
  }

  // 3. Rule-based Capture (Buffer)
  if (!cfg.smartCapture || lastUserMsg.length >= 15) {
     const toCapture = userTexts.filter(t => shouldCapture(t, { maxChars: cfg.captureMaxChars }));
     for (const text of toCapture.slice(0, 3)) {
       const category = detectCategory(text);
       const importance = (category === "entity" || category === "decision") ? 0.85 : 0.7;
       const promotion = workingMemory.add(text, importance, category);
       if (promotion.promoted) {
         const vector = await embeddings.embed(text);
         const summary = await generateMemorySummary(text, chatModel, logger);
         await db.store({ text, vector, importance, category, summary });
         tracer.traceStore(text, category, "rule-based");
       }
     }
  }
}

async function processFacts(
  facts: any[],
  db: MemoryDB,
  embeddings: Embeddings,
  graphDB: GraphDB,
  chat: ChatModel,
  tracer: MemoryTracer,
  logger: Logger,
) {
  const vectors = await embeddings.embedBatch(facts.map(f => f.text));
  const storedFacts: string[] = [];

  for (let i = 0; i < facts.length; i++) {
    const fact = facts[i];
    const vector = vectors[i];
    
    // Check for contradictions or duplicates
    const existing = await db.search(vector, 1, 0.95);
    if (existing.length > 0) continue;

    const entry = await db.store({
      text: fact.text,
      vector,
      importance: fact.importance,
      category: fact.category,
      summary: fact.summary || fact.text.slice(0, 100),
      emotionalTone: fact.emotionalTone || "neutral",
      emotionScore: fact.emotionScore || 0,
    });
    tracer.traceStore(fact.text, fact.category, entry.id);
    storedFacts.push(fact.text);
  }

  if (storedFacts.length > 0) {
    const graph = await extractGraphFromBatch(storedFacts, chat, TaskPriority.LOW, tracer, logger);
    await graphDB.modify(() => {
      for (const n of graph.nodes) graphDB.addNode(n);
      for (const e of graph.edges) graphDB.addEdge(e);
    });
  }
}
