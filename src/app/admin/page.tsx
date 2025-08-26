'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/auth-context';
import { auth } from '@/lib/firebase';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, LogOut, User, LayoutDashboard, FileText, Settings, Home } from 'lucide-react';
import Link from 'next/link';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';


const AdminPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/signin');
    }
  }, [user, loading, router]);

  const handleSignOut = async () => {
    await auth.signOut();
    router.push('/');
  };

  if (loading || !user) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <Loader2 className="h-16 w-16 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-muted/40">
      <aside className="hidden w-64 flex-col border-r bg-background p-4 sm:flex">
        <div className="flex items-center gap-2 pb-4 border-b">
          <Avatar>
            <AvatarImage src={user.photoURL ?? ''} />
            <AvatarFallback>{user.displayName?.charAt(0) ?? <User />}</AvatarFallback>
          </Avatar>
          <div className="flex flex-col">
            <span className="font-semibold">{user.displayName}</span>
            <span className="text-xs text-muted-foreground">{user.email}</span>
          </div>
        </div>
        <nav className="flex-1 space-y-2 py-4">
            <Button variant="ghost" className="w-full justify-start" asChild>
                <Link href="/"><Home className="mr-2 h-4 w-4" />Home</Link>
            </Button>
            <Button variant="secondary" className="w-full justify-start">
                <LayoutDashboard className="mr-2 h-4 w-4" />Dashboard
            </Button>
            <Button variant="ghost" className="w-full justify-start">
                <FileText className="mr-2 h-4 w-4" />Documents
            </Button>
            <Button variant="ghost" className="w-full justify-start">
                <Settings className="mr-2 h-4 w-4" />Settings
            </Button>
        </nav>
        <div className="mt-auto">
          <Button variant="outline" className="w-full" onClick={handleSignOut}>
            <LogOut className="mr-2 h-4 w-4" />
            Sign Out
          </Button>
        </div>
      </aside>
      <main className="flex-1 p-6">
        <Card>
          <CardHeader>
            <CardTitle>Admin Dashboard</CardTitle>
            <CardDescription>Welcome to your control panel, {user.displayName}.</CardDescription>
          </CardHeader>
          <CardContent>
            <p>This is a protected area. Here you can manage documents, users, and settings.</p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default AdminPage;
