'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Badge } from '@/components/ui/badge';

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface ExtractionFormProps {
  initialData: { [key: string]: string | number };
}

// Fields that are FIBO-aligned for the demo
const fiboAlignedFields = ['agreement_type', 'agreement_date', 'total_amount'];

export function ExtractionForm({ initialData }: ExtractionFormProps) {
  const { toast } = useToast();

  const formSchema = z.object(
    Object.keys(initialData).reduce((acc, key) => {
      acc[key] = z.string().min(1, 'This field is required.');
      return acc;
    }, {} as Record<string, z.ZodString>)
  );

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: Object.entries(initialData).reduce((acc, [key, value]) => {
      acc[key] = String(value);
      return acc;
    }, {} as Record<string, string>),
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    console.log('Submitting feedback:', values);
    toast({
      title: 'Feedback Submitted',
      description: 'Your corrections have been submitted successfully.',
    });
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          {Object.keys(initialData).map((key) => (
            <FormField
              key={key}
              control={form.control}
              name={key}
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="capitalize flex items-center">
                    {key.replace(/_/g, ' ')}
                    {fiboAlignedFields.includes(key) && (
                      <Badge variant="outline" className="ml-2 text-accent border-accent">FIBO</Badge>
                    )}
                  </FormLabel>
                  <FormControl>
                    <Input placeholder={`Enter ${key.replace(/_/g, ' ')}`} {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          ))}
        </div>
        <div className="flex justify-end pt-4">
            <Button type="submit">Submit Corrections</Button>
        </div>
      </form>
    </Form>
  );
}
