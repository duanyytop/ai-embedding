import {createOpenAI} from "@ai-sdk/openai";
import {cosineSimilarity, embedMany} from "ai";
import dotenv from "dotenv";

dotenv.config();

// Create an OpenAI provider instance with the API key
const openaiProvider = createOpenAI({
  apiKey: process.env.OPEN_API_KEY || "",
});

const embedding = async () => {
  const {embeddings} = await embedMany({
    model: openaiProvider.embedding("text-embedding-3-small"),
    values: ["sunny day at the beach", "rainy afternoon in the city"],
  });

  console.log(`cosine similarity: ${cosineSimilarity(embeddings[0], embeddings[1])}`);
}

embedding()
