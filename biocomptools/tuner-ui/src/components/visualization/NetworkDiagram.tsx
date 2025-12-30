import { useQuery } from '@tanstack/react-query'
import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronDown, Network } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { fetchDiagram } from '@/api/client'

export function NetworkDiagram() {
  const { data, isLoading } = useQuery({
    queryKey: ['diagram'],
    queryFn: () => fetchDiagram('all'),
    staleTime: Infinity,
  })

  return (
    <Collapsible.Root className="fixed bottom-0 left-0 right-0 bg-background border-t border-border">
      <Collapsible.Trigger asChild>
        <Button variant="ghost" className="w-full flex items-center gap-2 py-2 rounded-none">
          <Network className="h-4 w-4" />
          <span>Network Diagram</span>
          <ChevronDown className="h-4 w-4 ml-auto" />
        </Button>
      </Collapsible.Trigger>
      <Collapsible.Content className="p-4 max-h-96 overflow-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-48">
            <p className="text-muted-foreground">Loading diagram...</p>
          </div>
        ) : data?.svg ? (
          <div
            className="flex justify-center"
            dangerouslySetInnerHTML={{ __html: data.svg }}
          />
        ) : (
          <div className="text-muted-foreground text-center">No diagram available</div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  )
}
