import app from "./server.js";

const port = Number(process.env.RAG_BLOB_UPLOAD_PORT || 8503);
app.listen(port, "0.0.0.0");
