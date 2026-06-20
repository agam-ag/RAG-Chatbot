import * as dotenv from "dotenv";
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { GoogleGenAI } from "@google/genai";

// Gemini Client
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

// Delay helper
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Custom Gemini Embeddings
class GeminiEmbeddings {
  async embedQuery(text) {
    const response = await ai.models.embedContent({
      model: "gemini-embedding-001",
      contents: text,
    });

    return response.embeddings[0].values;
  }

  async embedDocuments(texts) {
    const vectors = [];

    for (let i = 0; i < texts.length; i++) {
      console.log(`Embedding ${i + 1}/${texts.length}`);

      try {
        const response = await ai.models.embedContent({
          model: "gemini-embedding-001",
          contents: texts[i],
        });

        vectors.push(response.embeddings[0].values);

        // Avoid free-tier rate limits
        await sleep(700);
      } catch (error) {
        console.log(`Retrying chunk ${i + 1}...`);

        await sleep(30000);

        const response = await ai.models.embedContent({
          model: "gemini-embedding-001",
          contents: texts[i],
        });

        vectors.push(response.embeddings[0].values);
      }
    }

    return vectors;
  }
}

async function indexDocument() {
  try {
    // Load PDF
    const pdfLoader = new PDFLoader("./Dsa.pdf");
    const rawDocs = await pdfLoader.load();

    console.log("PDF Loaded");

    // Chunking
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 100,
    });

    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

    console.log("Chunking Completed");
    console.log("Chunks:", chunkedDocs.length);

    // Embeddings
    const embeddings = new GeminiEmbeddings();

    const testVector = await embeddings.embedQuery(
      "What is Binary Search?"
    );

    console.log(
      "Embedding Dimension:",
      testVector.length
    );

    // Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const pineconeIndex = pinecone.Index(
      process.env.PINECONE_INDEX_NAME
    );

    console.log("Pinecone Configured");

    // Store in Pinecone
    await PineconeStore.fromDocuments(
      chunkedDocs,
      embeddings,
      {
        pineconeIndex,
        maxConcurrency: 1,
      }
    );

    console.log("✅ Data Stored Successfully");
  } catch (error) {
    console.error("ERROR:");
    console.error(error);
  }
}

indexDocument();