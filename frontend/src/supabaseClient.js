import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://dvkvquxcnojfkbasxrqt.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR2a3ZxdXhjbm9qZmtiYXN4cnF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0MDEzMzcsImV4cCI6MjA5MTk3NzMzN30.EktX_NWczkO00FG6S2iWZspnfuKzKo4Uj4b-Lf_x1Zk';

export const supabase = createClient(supabaseUrl, supabaseKey);
