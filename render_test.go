package main

import (
	"testing"

	"github.com/hajimehoshi/ebiten/v2/vector"
)

// TestWallPathCullsOffScreenTiles — стіни зібрані в ОДИН шлях, і відсікання мусить
// лишитись справжнім.
//
// Питання не в тому, скільки викликів vector ми зекономили — це арифметика, вона не
// може «зламатись». Ризик рівно один: разом із циклом малювання випадково зникне саме
// відсікання, і тоді в шлях полізуть усі тайли світу. Виглядатиме однаково, а платити
// будемо за 490 тайлів замість 218 — тобто тиха втрата того, заради чого пакетували.
//
// Тому міряємо КІЛЬКІСТЬ тайлів у шляху: світ 3400×1960 проти вікна 1700×980, тож при
// зумі 1.0 частина карти ЗОБОВʼЯЗАНА лишитись за кадром, а при 4× — тим паче.
func TestWallPathCullsOffScreenTiles(t *testing.T) {
	saved := cam
	defer func() { cam = saved }()

	total := 0
	for row := 0; row < boidMapH; row++ {
		for col := 0; col < boidMapW; col++ {
			if tileMap[row][col] {
				total++
			}
		}
	}
	if total == 0 {
		t.Fatal("на рівні немає стін — перевіряти нічого")
	}

	var p vector.Path
	cam.zoom = camZoomMin
	cam.cx, cam.cy = worldWidth/2, worldHeight/2
	at1 := buildWallPath(&p)

	cam.zoom = camZoomMax
	p.Reset()
	at4 := buildWallPath(&p)

	if at1 == 0 {
		t.Error("при зумі 1.0 у шлях не потрапило жодного тайла")
	}
	if at1 >= total {
		t.Errorf("при зумі 1.0 у шлях потрапили всі %d тайлів світу (%d) — відсікання не працює",
			total, at1)
	}
	if at4 >= at1 {
		t.Errorf("зум %.0f× не звузив набір тайлів: %d проти %d при 1.0", camZoomMax, at4, at1)
	}
}
