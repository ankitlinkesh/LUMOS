export default function Panel({
  id,
  number,
  title,
  description,
  children,
}: {
  id: string;
  number: number;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-slate-200 py-10 first:border-t-0 first:pt-0">
      <div className="mb-6 flex items-start gap-4">
        <span
          aria-hidden="true"
          className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-slate-900 text-lg font-bold text-white"
        >
          {number}
        </span>
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">{title}</h2>
          <p className="mt-1 max-w-3xl text-base text-slate-600 sm:text-lg">{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}
