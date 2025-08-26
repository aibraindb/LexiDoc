'use client';

import { useState, type DragEvent, type ChangeEvent } from 'react';
import { FileUp, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { suggestDocumentSchema } from '@/ai/flows/suggest-document-schema';

import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useToast } from '@/hooks/use-toast';
import { DocumentResults, type DocResult } from '@/components/document-results';

export function DocumentIntelligencePlatform() {
  const [isLoading, setIsLoading] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [result, setResult] = useState<DocResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    if (!file) return;

    if (file.type !== 'application/pdf') {
      toast({
        variant: 'destructive',
        title: 'Invalid File Type',
        description: 'Please upload a PDF document.',
      });
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);
    setProcessingStep('Reading document...');

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const dataUri = e.target?.result as string;
        setProcessingStep('Generating schema with AI...');
        
        const schemaSuggestion = await suggestDocumentSchema({ pdfDocumentDataUri: dataUri });
        
        setProcessingStep('Classifying document and extracting data...');
        // Simulate other backend processes
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        const mockResult: DocResult = {
          fileName: file.name,
          fileSize: file.size,
          classification: {
            type: 'Lease Agreement',
            confidence: 0.91,
            rationale: ["hit: 'Lease Agreement'", "hit: 'DocuSign Envelope ID'"],
          },
          extractedData: {
            agreement_type: 'Lease Agreement',
            agreement_date: '12/15/2023',
            lessor_name: 'Property Management Inc.',
            lessee_name: 'John Doe',
            total_amount: '2500.00',
            statement_period: 'N/A'
          },
          schema: schemaSuggestion.suggestedSchema,
          missingDocs: {
            required: ['bank_statement', 'invoice'],
            message: 'To complete the loan underwriting package, please also provide the last 2 months of bank statements and any outstanding invoices.',
          },
        };

        setResult(mockResult);
      } catch (err) {
        console.error(err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
        setError(`Failed to process document. ${errorMessage}`);
        toast({
          variant: 'destructive',
          title: 'Processing Failed',
          description: errorMessage,
        });
      } finally {
        setIsLoading(false);
        setProcessingStep('');
      }
    };
    reader.onerror = () => {
      setError('Failed to read file.');
      toast({
        variant: 'destructive',
        title: 'File Read Error',
        description: 'Could not read the selected file.',
      });
      setIsLoading(false);
    };
    reader.readAsDataURL(file);
  };
  
  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  const handleDragEvents = (e: DragEvent<HTMLDivElement>, dragging: boolean) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(dragging);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    handleDragEvents(e, false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const uploaderContent = (
    <div
      onDragEnter={(e) => handleDragEvents(e, true)}
      onDragLeave={(e) => handleDragEvents(e, false)}
      onDragOver={(e) => handleDragEvents(e, true)}
      onDrop={handleDrop}
      className={cn(
        'border-2 border-dashed rounded-lg p-12 text-center transition-colors duration-200 ease-in-out',
        isDragging ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'
      )}
    >
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="flex flex-col items-center space-y-4">
          <FileUp className="w-16 h-16 text-muted-foreground" />
          <p className="text-lg font-semibold text-foreground">Drag & drop your PDF here</p>
          <p className="text-muted-foreground">or click to browse</p>
        </div>
        <Input id="file-upload" type="file" className="hidden" onChange={handleFileChange} accept=".pdf" disabled={isLoading} />
      </label>
    </div>
  );

  const loadingContent = (
    <div className="flex flex-col items-center justify-center p-12">
      <Loader2 className="w-16 h-16 animate-spin text-primary" />
      <p className="mt-4 text-lg font-semibold">{processingStep}</p>
      <p className="text-muted-foreground">Please wait while we analyze your document...</p>
    </div>
  );

  return (
    <Card className="w-full max-w-4xl mx-auto shadow-xl">
      <CardContent className="p-6">
        {isLoading 
          ? loadingContent
          : result 
          ? <DocumentResults result={result} onReset={handleReset} /> 
          : uploaderContent
        }
        {error && !isLoading && (
          <div className="mt-4 text-center text-destructive">
            <p>{error}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
