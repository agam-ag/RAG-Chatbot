import * as dotenv from "dotenv";
dotenv.config();

import { GoogleGenAI } from "@google/genai";
import { Pinecone } from "@pinecone-database/pinecone";
import readlineSync from "readline-sync";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

const History = [];

async function transformQuery(question) {
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
    Only output the rewritten question and nothing else.
      `,
    },
  });

  History.pop();

  return response.text;
}

async function chatting(question) {
  
  const queries = await transformQuery(question)
    // Create query embedding
  const embeddingResponse = await ai.models.embedContent({
    model: "gemini-embedding-001",
    contents: queries,
  });

  const queryVector = embeddingResponse.embeddings[0].values;

  // Connect to Pinecone
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  // Search
  const searchResults = await pineconeIndex.query({
    vector: queryVector,
    topK: 5,
    includeMetadata: true,
  });

  // Fetch the text part from the metadata and create the context for LLM
  const context = searchResults.matches
    .map((match) => match.metadata.text)
    .join("\n\n---\n\n");

  // Gemini
  History.push({
    role: "user",
    parts: [{ text: queries }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: ${context}
      `,
    },
  });

  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });

  console.log("\n");
  console.log(response.text);
}

async function main() {
  while (true) {
    const userProblem = readlineSync.question("\nAsk me anything --> ");

    await chatting(userProblem);
  }
}

main();
