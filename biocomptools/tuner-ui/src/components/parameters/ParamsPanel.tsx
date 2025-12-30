import * as Accordion from '@radix-ui/react-accordion'
import { ChevronDown } from 'lucide-react'
import { useTunerStore } from '@/store/tunerStore'
import { RatioGroup } from './RatioGroup'
import { ParamInput } from './ParamInput'

export function ParamsPanel() {
  const { paramGroups } = useTunerStore()

  const ratioGroups = paramGroups.filter((g) => g.is_ratio_group)
  const otherGroups = paramGroups.filter((g) => !g.is_ratio_group)

  return (
    <div className="flex-1 overflow-auto">
      <Accordion.Root type="multiple" defaultValue={['ratios']} className="w-full">
        <Accordion.Item value="ratios">
          <Accordion.Trigger className="flex w-full items-center justify-between py-2 px-4 text-sm font-medium hover:bg-accent">
            <span>Ratios ({ratioGroups.reduce((acc, g) => acc + g.params.length, 0)})</span>
            <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200 [&[data-state=open]>svg]:rotate-180" />
          </Accordion.Trigger>
          <Accordion.Content className="overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down">
            <div className="px-2 pb-2">
              {ratioGroups.map((group) => (
                <RatioGroup key={group.group_id} group={group} />
              ))}
            </div>
          </Accordion.Content>
        </Accordion.Item>

        {otherGroups.map((group) => (
          <Accordion.Item key={group.group_id} value={group.group_id}>
            <Accordion.Trigger className="flex w-full items-center justify-between py-2 px-4 text-sm font-medium hover:bg-accent">
              <span>
                {group.group_name} ({group.params.length})
              </span>
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
            </Accordion.Trigger>
            <Accordion.Content className="overflow-hidden">
              <div className="px-4 pb-2 space-y-2">
                {group.params.map((param) => (
                  <ParamInput key={param.path} param={param} />
                ))}
              </div>
            </Accordion.Content>
          </Accordion.Item>
        ))}
      </Accordion.Root>
    </div>
  )
}
