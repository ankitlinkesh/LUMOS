type Variant = "neutral" | "red" | "green" | "amber" | "blue";

const VARIANT_CLASSES: Record<Variant, string> = {
  neutral: "bg-slate-100 text-slate-700 border-slate-300",
  red: "bg-red-50 text-red-800 border-red-300",
  green: "bg-emerald-50 text-emerald-800 border-emerald-300",
  amber: "bg-amber-50 text-amber-800 border-amber-300",
  blue: "bg-blue-50 text-blue-800 border-blue-300",
};

export default function Badge({
  children,
  variant = "neutral",
  title,
}: {
  children: React.ReactNode;
  variant?: Variant;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-sm font-medium whitespace-nowrap ${VARIANT_CLASSES[variant]}`}
    >
      {children}
    </span>
  );
}
