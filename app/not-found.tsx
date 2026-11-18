import Link from 'next/link';

export default function NotFound() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-[#F8FAFC] px-6 py-16 text-[#0F172A]">
      <div className="max-w-md w-full bg-white border border-[#E2E8F0] rounded-lg p-6 space-y-4">
        <div className="text-xs font-mono text-[#64748B]">HTTP 404 · ROUTE NOT FOUND</div>
        <h1 className="text-xl font-semibold text-[#0F172A]">Page Not Found</h1>
        <p className="text-sm text-[#475569]">
          The requested path does not exist in this QH4-FieldPermutation workbench.
        </p>
        <div className="pt-1">
          <Link
            href="/"
            className="inline-flex items-center px-4 py-2 bg-[#0F4C81] hover:bg-[#0C3E6B] text-white text-xs font-medium rounded-md"
          >
            Return to Workbench
          </Link>
        </div>
      </div>
    </main>
  );
}
