# **App Name**: DocIntelliBank

## Core Features:

- AI-Powered Schema Suggestion: Suggest a schema variant based on the document content. A tool LLM is employed, determining when the automatically suggested information can improve the result.
- PDF Document Ingestion: Ingest and process one or more PDF documents independently, then aggregate results.
- Document Classification: Classify each document type (bank statement, invoice, etc.) with confidence scores and rationale, using both rule-based and (future) ML-based classifiers.
- Key-Value Extraction: Extract key-value pairs, normalize units, and parse money/date fields based on a defined schema (FIBO-aligned).
- Vector Embedding and Storage: Store document chunks, field values, and headers in an in-memory FAISS vector store for semantic search.
- Missing Document Detection: Detect missing documents based on pre-defined required sets and generate a customer request message.
- Feedback and Correction Submission: Provide an API endpoint for submitting corrections (field overrides, type corrections) to improve the system's accuracy.

## Style Guidelines:

- Primary color: Deep blue (#3F51B5) to evoke trust and stability, reflecting the financial context.
- Background color: Light gray (#F0F2F5), a soft neutral to ensure readability and focus on content.
- Accent color: Teal (#009688), which is analogous to blue, yet different enough in brightness and saturation to signal important actions.
- Body and headline font: 'Inter', a grotesque-style sans-serif for a modern and neutral appearance. Suitable for both headlines and body text, and implies a machined, objective feel which will add credibility.
- Use crisp, professional icons related to document types and financial data.
- Maintain a clean and organized layout to ensure easy readability and navigation through financial documents.
- Employ subtle transitions and loading animations to provide a smooth user experience.