import express from "express";
import { del } from "@vercel/blob";
import { handleUpload } from "@vercel/blob/client";

const app = express();
const maxUploadBytes = 2 * 1024 ** 3;
const privateToken = process.env.RAG_UPLOAD_READ_WRITE_TOKEN;
const allowedExtensions = new Set([".pdf", ".txt", ".md", ".csv", ".json", ".py", ".html"]);
const allowedContentTypes = [
  "application/pdf",
  "text/plain",
  "text/markdown",
  "text/csv",
  "application/json",
  "text/html",
  "text/x-python",
  "application/octet-stream",
];

app.use(express.json({ limit: "1mb" }));

function isAllowedPathname(pathname) {
  const segments = pathname.split("/");
  if (segments.length !== 2 || segments[0] !== "rag-uploads") return false;
  const name = segments[1];
  if (!name || name.includes("..") || name.includes("\\")) return false;
  return allowedExtensions.has(name.slice(name.lastIndexOf(".")).toLowerCase());
}

app.post("/api/blob-upload", async (request, response) => {
  if (!privateToken) {
    return response.status(503).json({ detail: "Private Blob storage is not connected to this environment." });
  }

  try {
    const result = await handleUpload({
      body: request.body,
      request,
      token: privateToken,
      onBeforeGenerateToken: async (pathname) => {
        if (!isAllowedPathname(pathname)) throw new Error("Unsupported upload path or file type.");
        return {
          allowedContentTypes,
          maximumSizeInBytes: maxUploadBytes,
          validUntil: Date.now() + 24 * 60 * 60 * 1000,
          addRandomSuffix: false,
          ttlDays: 30,
        };
      },
    });
    return response.status(200).json(result);
  } catch {
    return response.status(400).json({ detail: "Could not authorize this private Blob upload." });
  }
});

app.delete("/api/blob-upload", async (request, response) => {
  if (!privateToken) {
    return response.status(503).json({ detail: "Private Blob storage is not connected to this environment." });
  }

  const pathname = request.body?.pathname;
  if (typeof pathname !== "string" || !isAllowedPathname(pathname)) {
    return response.status(400).json({ detail: "Invalid private upload path." });
  }

  try {
    await del(pathname, { access: "private", token: privateToken });
    return response.status(200).json({ deleted: true });
  } catch {
    return response.status(400).json({ detail: "Could not delete this private upload." });
  }
});

app.use((error, request, response, next) => {
  response.status(400).json({ detail: "Invalid Blob request." });
});

export default app;
