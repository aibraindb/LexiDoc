// This file is machine-generated - do not edit!

'use server';

/**
 * @fileOverview This file defines a Genkit flow for suggesting a document schema based on the document's content.
 *
 * The flow takes a PDF document as input and uses an LLM to identify relevant fields and their corresponding data types,
 * so that users can quickly and accurately extract information from the document.
 *
 * - suggestDocumentSchema - A function that initiates the document schema suggestion flow.
 * - SuggestDocumentSchemaInput - The input type for the suggestDocumentSchema function, which includes the PDF document as a data URI.
 * - SuggestDocumentSchemaOutput - The output type for the suggestDocumentSchema function, which contains the suggested schema as a JSON string.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const SuggestDocumentSchemaInputSchema = z.object({
  pdfDocumentDataUri: z
    .string()
    .describe(
      'The PDF document content as a data URI that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.' 
    ),
});
export type SuggestDocumentSchemaInput = z.infer<typeof SuggestDocumentSchemaInputSchema>;

const SuggestDocumentSchemaOutputSchema = z.object({
  suggestedSchema: z.string().describe('The suggested document schema as a JSON string.'),
});
export type SuggestDocumentSchemaOutput = z.infer<typeof SuggestDocumentSchemaOutputSchema>;

export async function suggestDocumentSchema(input: SuggestDocumentSchemaInput): Promise<SuggestDocumentSchemaOutput> {
  return suggestDocumentSchemaFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestDocumentSchemaPrompt',
  input: {schema: SuggestDocumentSchemaInputSchema},
  output: {schema: SuggestDocumentSchemaOutputSchema},
  prompt: `You are an expert document schema generator. Given the content of a PDF document, you will identify the relevant fields and their corresponding data types, and generate a JSON schema that can be used to extract information from the document.

  Document Content: {{media url=pdfDocumentDataUri}}

  Provide only a JSON schema as output. Do not include any other text or explanations.`,
});

const suggestDocumentSchemaFlow = ai.defineFlow(
  {
    name: 'suggestDocumentSchemaFlow',
    inputSchema: SuggestDocumentSchemaInputSchema,
    outputSchema: SuggestDocumentSchemaOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
