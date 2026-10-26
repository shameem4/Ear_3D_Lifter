#!/bin/bash
heads=(pp1 pp2 pp3 pp4 pp5 pp6 pp8 pp9 pp10 pp11 pp12 pp16 pp19 pp20 pp21 pp22 pp23 pp29 pp30 pp31 pp32 pp33 pp40 pp41 pp44 pp45 pp46 pp47 pp48 pp49 pp55 pp57 pp58 pp59 pp60 pp61 pp62 pp63 pp66 pp67 pp68 pp69 pp70 pp71 pp72 pp73 pp76 pp77 pp78 pp80)
for i in 0 1 2 3; do
  /home/shameem/.conda/envs/mocap/bin/python /var/tmp/claude/ear3d/2026-10-25-face-wrapkeys/wk2.py $(for j in ${!heads[@]}; do [ $((j%4)) = $i ] && echo ${heads[j]}; done) > /var/tmp/claude/ear3d/2026-10-25-face-wrapkeys/s$i.log 2>&1 &
done
wait
