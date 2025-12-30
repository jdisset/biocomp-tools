import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getLossColor(total: number): string {
  if (total < 0.1) return 'text-loss-good'
  if (total < 0.3) return 'text-loss-medium'
  return 'text-loss-bad'
}

export function formatNumber(value: number, decimals = 4): string {
  return value.toFixed(decimals)
}
