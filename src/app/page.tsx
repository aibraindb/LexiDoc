'use client';

import { DocumentIntelligencePlatform } from '@/components/document-intelligence-platform';
import { useAuth } from '@/context/auth-context';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { LogOut, User } from 'lucide-react';
import { auth } from '@/lib/firebase';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';


export default function Home() {
  const { user, loading } = useAuth();

  const handleSignOut = async () => {
    await auth.signOut();
  };

  return (
    <div className="container mx-auto px-4 py-8 md:py-12">
      <header className="flex justify-between items-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-primary font-headline">InteliBank</h1>
        <div className="flex items-center gap-4">
          {loading ? (
            <div className="h-10 w-24 animate-pulse rounded-md bg-muted" />
          ) : user ? (
            <>
               <Button asChild variant="outline">
                <Link href="/admin">Admin Dashboard</Link>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger>
                  <Avatar>
                    <AvatarImage src={user.photoURL ?? ''} alt={user.displayName ?? 'User'} />
                    <AvatarFallback>
                      <User />
                    </AvatarFallback>
                  </Avatar>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>{user.displayName}</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleSignOut}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          ) : (
            <Button asChild>
              <Link href="/auth/signin">Sign In</Link>
            </Button>
          )}
        </div>
      </header>
      <div className="text-center mb-12">
        <p className="mt-4 text-lg text-muted-foreground max-w-3xl mx-auto">
          Intelligent Document Processing for Financial Institutions. Upload a bank document (e.g., invoice, loan agreement, bank statement) to classify it, extract key information, and get an AI-suggested data schema.
        </p>
      </div>
      <DocumentIntelligencePlatform />
    </div>
  );
}
