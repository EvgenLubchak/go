package main

import (
	"fmt"
	"os"
	"sort"
	"testing"
	"time"
)

// [ПРОБА] Тимчасовий профіль: куди йде час ОДНОГО ТІКУ ЛОГІКИ.
//
// Міряємо саме логіку, бо FPS і TPS — різні речі, і плутати їх ми вже вчились:
// FPS = частота Draw, TPS = частота Update. Якщо один тік логіки не влазить у бюджет
// 1/120 с = 8.33 мс, то оптимізувати рендер безглуздо — вузьке місце не там.
//
// Дві комірки: k=1 — теперішній ростер, k=2 — той, на якому було видно просадку.
//
// Запуск: BOIDS_PROFILE=1 go test -run TestTickProfile -v

type stageTime struct {
	name string
	ns   int64
}

func profileCell(k, warmup, window int) (stages []stageTime, units int, totalNs int64) {
	type keep struct{ count, resp int }
	saved := make([]keep, len(unitRoster))
	for i := range unitRoster {
		saved[i] = keep{unitRoster[i].Count, unitRoster[i].Respawns}
		unitRoster[i].Count *= k
		unitRoster[i].Respawns = -1 // стала популяція, інакше знаменник пливе
	}
	defer func() {
		for i := range unitRoster {
			unitRoster[i].Count = saved[i].count
			unitRoster[i].Respawns = saved[i].resp
		}
	}()

	g := newBenchGame()
	drive := benchDriver(true, true)
	for t := 0; t < warmup; t++ {
		g.tickHeadless(drive, true)
	}
	units = len(g.units)

	acc := map[string]int64{}
	timeIt := func(name string, f func()) {
		s := time.Now()
		f()
		acc[name] += time.Since(s).Nanoseconds()
	}

	start := time.Now()
	for t := 0; t < window; t++ {
		g.tick++
		timeIt("drive+гравець", func() { drive(g); g.updatePlayer() })
		timeIt("flow-field", func() { g.updateFlowFields() })
		timeIt("boidMap", func() { g.updateBoidMap() })
		timeIt("calcAcceleration", func() { g.calcAcceleration() })
		timeIt("trainBrains", func() { g.trainBrains() })
		timeIt("metrics", func() { g.metrics.collect(g) })
		timeIt("updateUnits", func() { g.updateUnits() })
		timeIt("бій", func() {
			g.resolveImpacts()
			for i := range g.units {
				g.pushOffPlayer(&g.units[i])
			}
			g.handleDeadUnits()
		})
		// Анімація окремо: ці функції викликає updateUnits, тож їхній час УЖЕ
		// врахований вище. Тут заміряємо їх ще раз на тих самих юнітах, щоб побачити
		// ЧАСТКУ анімації всередині updateUnits — це і є питання «ворс/щупальце/руки».
		timeIt("  ⤷ ворс", func() {
			for i := range g.units {
				updateFur(&g.units[i])
			}
		})
		timeIt("  ⤷ щупальце", func() {
			for i := range g.units {
				updateTentacle(&g.units[i])
			}
		})
		timeIt("  ⤷ кінцівки", func() {
			for i := range g.units {
				updateLimbs(&g.units[i])
			}
		})
		timeIt("  ⤷ кульки", func() {
			for i := range g.units {
				updateBalls(&g.units[i])
			}
		})
		timeIt("  ⤷ тіло", func() {
			for i := range g.units {
				updateBody(&g.units[i])
			}
		})
	}
	totalNs = time.Since(start).Nanoseconds() / int64(window)

	for n, v := range acc {
		stages = append(stages, stageTime{n, v / int64(window)})
	}
	sort.Slice(stages, func(i, j int) bool { return stages[i].ns > stages[j].ns })
	return stages, units, totalNs
}

func TestTickProfile(t *testing.T) {
	if os.Getenv("BOIDS_PROFILE") == "" {
		t.Skip("профіль тіку: BOIDS_PROFILE=1")
	}
	const warmup, window = 3000, 4000
	budget := 1000.0 / 120.0 // мс на тік при 120 TPS

	for _, k := range []int{1, 2} {
		stages, units, total := profileCell(k, warmup, window)
		fmt.Printf("\n=== %d юнітів ===  тік разом: %.3f мс   бюджет 120 TPS: %.2f мс\n",
			units, float64(total)/1e6, budget)
		for _, s := range stages {
			ms := float64(s.ns) / 1e6
			fmt.Printf("  %-18s %7.3f мс  %5.1f%% бюджету\n", s.name, ms, ms/budget*100)
		}
	}
}
