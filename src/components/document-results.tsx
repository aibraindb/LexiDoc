'use client';

import { KeyRound, FileJson, FileText, FileWarning, BadgeCheck, ListTree } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { ExtractionForm } from '@/components/extraction-form';

export interface DocResult {
  fileName: string;
  fileSize: number;
  classification: {
    type: string;
    confidence: number;
    rationale: string[];
  };
  extractedData: { [key: string]: string | number };
  schema: string;
  missingDocs: {
    required: string[];
    message: string;
  } | null;
}

interface DocumentResultsProps {
  result: DocResult;
  onReset: () => void;
}

function formatBytes(bytes: number, decimals = 2) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function DocumentResults({ result, onReset }: DocumentResultsProps) {
  let parsedSchema = "{}";
  try {
    parsedSchema = JSON.stringify(JSON.parse(result.schema), null, 2);
  } catch (e) {
    parsedSchema = "Invalid JSON schema received from AI.";
  }

  return (
    <div>
      <CardHeader className="p-0 mb-6">
        <CardTitle className="text-2xl">{result.fileName}</CardTitle>
        <CardDescription>{formatBytes(result.fileSize)}</CardDescription>
      </CardHeader>

      {result.missingDocs && (
        <Alert variant="destructive" className="mb-6">
          <FileWarning className="h-4 w-4" />
          <AlertTitle>Missing Documents Detected</AlertTitle>
          <AlertDescription>
            {result.missingDocs.message} Required: {result.missingDocs.required.join(', ')}.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="extraction" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="extraction"><KeyRound className="mr-2 h-4 w-4" />Extracted Data</TabsTrigger>
          <TabsTrigger value="schema"><FileJson className="mr-2 h-4 w-4" />Suggested Schema</TabsTrigger>
          <TabsTrigger value="classification"><FileText className="mr-2 h-4 w-4" />Classification</TabsTrigger>
        </TabsList>
        <TabsContent value="extraction" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Fields</CardTitle>
              <CardDescription>Review and correct the extracted information. Fields marked with (FIBO) are aligned with Financial Industry Business Ontology standards.</CardDescription>
            </CardHeader>
            <CardContent>
              <ExtractionForm initialData={result.extractedData} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="schema" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>AI-Suggested Schema</CardTitle>
              <CardDescription>This is the JSON schema suggested by the AI based on the document's content.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-muted rounded-md p-4 max-h-96 overflow-auto">
                <pre><code className="text-sm font-code">{parsedSchema}</code></pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="classification" className="mt-4">
           <Card>
            <CardHeader>
              <CardTitle>Document Classification</CardTitle>
              <CardDescription>Analysis of the document type and structure.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-center space-x-4">
                    <BadgeCheck className="h-10 w-10 text-accent" />
                    <div>
                        <p className="font-semibold text-lg">{result.classification.type}</p>
                        <p className="text-sm text-muted-foreground">Confidence: <span className="font-bold text-accent">{(result.classification.confidence * 100).toFixed(0)}%</span></p>
                    </div>
                </div>
                <div>
                    <h4 className="font-semibold mb-2 flex items-center"><ListTree className="mr-2 h-4 w-4"/>Rationale:</h4>
                    <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                        {result.classification.rationale.map((reason, index) => (
                            <li key={index}>{reason}</li>
                        ))}
                    </ul>
                </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      <div className="mt-6 flex justify-end">
        <Button onClick={onReset}>Process Another Document</Button>
      </div>
    </div>
  );
}
