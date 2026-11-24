# NX-200 Warehouse Robot: Product Specification (fictional sample)

## Overview

The NX-200 is an autonomous mobile robot that moves shelves and totes inside warehouses. It navigates using two LiDAR sensors and a front-facing depth camera, and it does not need floor markers or magnetic tape.

## Specifications

- Payload capacity: 600 kg
- Top speed: 1.8 m/s unloaded, 1.2 m/s at full load
- Battery: 48 V lithium iron phosphate, 8 hours of typical operation per charge
- Charging: autonomous docking; 0 to 80% charge in 55 minutes
- Dimensions: 1,100 × 800 × 350 mm
- Operating temperature: 0 °C to 40 °C
- Safety: certified to ISO 3691-4; emergency stop buttons on all four sides

## Fleet management

Up to 150 robots can be coordinated by one NX Fleet Manager server. The fleet manager assigns tasks, plans routes to avoid congestion and sends robots to charge when their battery drops below 20%. It integrates with warehouse management systems through a REST API and webhooks.

## Maintenance

Wheels and LiDAR covers should be cleaned weekly. Drive wheels should be replaced every 5,000 operating hours. Firmware updates are delivered over the air and installed only while the robot is docked.

## Warranty

The NX-200 comes with a 2-year standard warranty covering parts and labour. Batteries are covered for 3 years or 3,000 charge cycles, whichever comes first. Damage caused by operating outside the specified temperature range is not covered.
