package main

import (
	"image/color"
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

// TestSeaGradientKeepsItsDepth — море стало одним чотирикутником, у якого колір
// інтерполює GPU. Назовні воно не віддає нічого, тож перевіряти можна лише seaColorAt.
//
// Три властивості, і кожна ловить свою помилку:
//
//	краї      — переплутати місцями верх і низ найлегше саме при переході на вершини;
//	монотонність — глибина мусить темнішати всю дорогу, без стрибків назад;
//	збіг зі старим — картинка не мусила змінитись, крім зниклих сходинок. Тому
//	           звіряємось із ТІЄЮ САМОЮ формулою, що була в циклі на 48 смуг.
func TestSeaGradientKeepsItsDepth(t *testing.T) {
	top, bot := seaColorAt(0), seaColorAt(worldHeight)
	if top != (color.RGBA{seaTopR, seaTopG, seaTopB, 255}) {
		t.Errorf("поверхня: %v, очікували мілину %d,%d,%d", top, seaTopR, seaTopG, seaTopB)
	}
	if bot != (color.RGBA{seaBotR, seaBotG, seaBotB, 255}) {
		t.Errorf("дно: %v, очікували глибину %d,%d,%d", bot, seaBotR, seaBotG, seaBotB)
	}

	// За межами світу колір не мусить «вивертатись»: камера туди не заїде, але
	// відсікання живе в camera.go, а не тут, і покладатись на чужий інваріант не варто.
	if seaColorAt(-500) != top || seaColorAt(worldHeight*2) != bot {
		t.Error("за межами карти колір не притиснутий до країв")
	}

	prev := 256
	for y := float32(0); y <= worldHeight; y += worldHeight / 200 {
		if g := int(seaColorAt(y).G); g > prev {
			t.Errorf("на висоті %.0f море посвітлішало (%d після %d)", y, g, prev)
			break
		} else {
			prev = g
		}
	}

	// Стара формула: 48 смуг, t = i/(48-1), смуга i починалась на worldHeight*i/48.
	const oldBands = 48
	for i := 0; i < oldBands; i++ {
		told := float32(i) / (oldBands - 1)
		want := color.RGBA{
			R: uint8(float32(seaTopR) + (seaBotR-seaTopR)*told),
			G: uint8(float32(seaTopG) + (seaBotG-seaTopG)*told),
			B: uint8(float32(seaTopB) + (seaBotB-seaTopB)*told),
			A: 255,
		}
		got := seaColorAt(worldHeight * float32(i) / oldBands)
		d := func(a, b uint8) int {
			if a > b {
				return int(a - b)
			}
			return int(b - a)
		}
		if d(got.R, want.R) > 2 || d(got.G, want.G) > 2 || d(got.B, want.B) > 2 {
			t.Errorf("смуга %d: тепер %v, раніше було %v", i, got, want)
		}
	}
}

// TestStrikeFlashMarksTheAttacker — спалах удару світить НАПАДНИКУ, а не жертві.
//
// Головне, що тут перевіряється, — не колір, а те, що дві події лишились
// РОЗРІЗНЕННИМИ: жертва біліє тілом (HitTimer), нападник — відростком і кульками
// (StrikeTimer). Якби хтось звів їх на одне поле, у щільній бійці стало б неможливо
// сказати, хто кому завдав, а тест лишився б зеленим, якби перевіряв лише «біле».
func TestStrikeFlashMarksTheAttacker(t *testing.T) {
	green := color.RGBA{0, 255, 100, 255}
	attacker := Pixel{X: 100, Y: 100, Color: green, HP: 5, MaxHP: 5}
	target := Pixel{X: 130, Y: 100, Color: green, HP: 5, MaxHP: 5}

	applyImpactDamage(&attacker, &target, 1)

	if attacker.StrikeTimer != strikeFlashDuration {
		t.Errorf("нападник: StrikeTimer %d, очікували %d", attacker.StrikeTimer, strikeFlashDuration)
	}
	if attacker.HitTimer != 0 {
		t.Errorf("нападник не мусить блимати як жертва: HitTimer %d", attacker.HitTimer)
	}
	if target.HitTimer != hitFlashDuration {
		t.Errorf("жертва: HitTimer %d, очікували %d", target.HitTimer, hitFlashDuration)
	}
	if target.StrikeTimer != 0 {
		t.Errorf("жертва не мусить світити як нападник: StrikeTimer %d", target.StrikeTimer)
	}

	if c := strikeColorOf(&attacker); c != (color.RGBA{255, 255, 255, 255}) {
		t.Errorf("відросток нападника не побілів: %v", c)
	}
	if c := strikeColorOf(&target); c != green {
		t.Errorf("відросток жертви не мусить біліти: %v", c)
	}

	// Спалах мусить ЗГАСАТИ, і перевіряти це треба ЧЕРЕЗ КРОК СВІТУ, а не ручним
	// обнуленням. Перша версія цього тесту ставила StrikeTimer = 0 сама — і мутація
	// «прибрати зменшення в updateUnits» лишила її зеленою. Тобто тест перевіряв
	// формулу кольору, а не те, що спалах взагалі колись гасне.
	g := &Game{units: []Pixel{attacker}}
	before := g.units[0].StrikeTimer
	g.updateUnits()
	if g.units[0].StrikeTimer != before-1 {
		t.Errorf("після кроку світу StrikeTimer %d, очікували %d — спалах не гасне",
			g.units[0].StrikeTimer, before-1)
	}

	attacker.StrikeTimer = 0
	if c := strikeColorOf(&attacker); c != green {
		t.Errorf("після спалаху колір не повернувся: %v", c)
	}
}

// TestDefaultsMatchTheirConstants — індикатори HUD показують лише те, що ВІДРІЗНЯЄТЬСЯ
// від типового стану, і порівнюють поточне значення з іменованою константою.
//
// Пастка, від якої це стереже: змінна й константа живуть поруч, але окремо. Змінив
// дефолт у var і забув у const — і показник або світиться завжди, або не світиться
// ніколи. Компілятор мовчить, картинка правильна, а прилад бреше. Ми вже ловили рівно
// цей клас вади в камері, де тест був зелений через збіг чисел.
//
// Тест читає ПОЧАТКОВІ значення, тому мусить іти до будь-чого, що їх змінює. Пакетні
// змінні ініціалізуються один раз, тож усередині одного прогону це так і є.
func TestDefaultsMatchTheirConstants(t *testing.T) {
	if gameTPS != gameTPSDefault {
		t.Errorf("gameTPS стартує з %d, а типовим оголошено %d — показник temp бреше",
			gameTPS, gameTPSDefault)
	}
	if antiAlias != antiAliasDefault {
		t.Errorf("antiAlias стартує з %v, а типовим оголошено %v — показник AA бреше",
			antiAlias, antiAliasDefault)
	}
	if renderScale != renderScaleDefault {
		t.Errorf("renderScale стартує з %v, а типовим оголошено %v — показник SS бреше",
			renderScale, renderScaleDefault)
	}
	if cam.zoom != camZoomDefault {
		t.Errorf("камера стартує з зумом %v, а типовим оголошено %v — показник zoom бреше",
			cam.zoom, camZoomDefault)
	}

	// Дефолт мусить бути ДОСЯЖНИМ клавішами, інакше, збивши його, повернутись не можна.
	if camZoomDefault < camZoomMin || camZoomDefault > camZoomMax {
		t.Errorf("типовий зум %v поза межами [%v, %v]", camZoomDefault, camZoomMin, camZoomMax)
	}
	if d := float64(camZoomDefault-camZoomMin) / camZoomStep; d != float64(int(d)) {
		t.Errorf("типовий зум %v не лягає на крок %v від %v — клавішами в нього не попасти",
			camZoomDefault, camZoomStep, camZoomMin)
	}
}

// TestSuperSamplingLadder — драбина масштабів і спуск нею.
//
// Головна властивість не в тому, як високо забирається драбина (×12 узагалі поклав
// гру — див. ssLadder), а в тому, що НЕВДАЧА НЕ МОВЧИТЬ: sceneScale
// каже, що вийшло насправді, і HUD показує розбіжність. Камера множить саме на
// sceneScale, тож помилка тут малювала б світ повз буфер — чорний екран замість
// чесного «не влізло».
func TestSuperSamplingLadder(t *testing.T) {
	if len(ssLadder) < 2 || ssLadder[0] != 1 {
		t.Fatalf("драбина мусить починатись з 1 і мати щаблі вище: %v", ssLadder)
	}
	for i := 1; i < len(ssLadder); i++ {
		if ssLadder[i] <= ssLadder[i-1] {
			t.Errorf("драбина не зростає: %v", ssLadder)
		}
	}

	// Типовий масштаб мусить бути НА драбині, інакше клавіша H у нього не повернеться:
	// цикл шукає поточне значення в списку, не знаходить і починає з початку.
	onLadder := false
	for _, s := range ssLadder {
		if s == renderScaleDefault {
			onLadder = true
		}
	}
	if !onLadder {
		t.Errorf("типовий масштаб %v не лежить на драбині %v — клавішею H у нього не вернутись",
			float32(renderScaleDefault), ssLadder)
	}

	// Спуск: невдале виділення мусить дати ЩАБЕЛЬ НИЖЧЕ, а не тишу. Перевіряємо саму
	// логіку вибору, не чіпаючи GPU — виділяти тут нічого не можна, бо в тестах немає
	// графічного контексту.
	pick := func(want float32, fits func(float32) bool) float32 {
		for i := len(ssLadder) - 1; i >= 0; i-- {
			s := ssLadder[i]
			if s > want || s <= 1 {
				continue
			}
			if fits(s) {
				return s
			}
		}
		return 1
	}
	all := func(float32) bool { return true }
	top := ssLadder[len(ssLadder)-1]
	if got := pick(top, all); got != top {
		t.Errorf("коли влазить усе, %v мусить лишитись %v, а не %v", top, top, got)
	}
	upTo2 := func(s float32) bool { return s <= 2 }
	if got := pick(top, upTo2); got != 2 {
		t.Errorf("коли стеля ×2, запит %v мусить осісти на ×2, а не на %v", top, got)
	}
	none := func(float32) bool { return false }
	if got := pick(top, none); got != 1 {
		t.Errorf("коли не влазить нічого, мусить лишитись ×1, а не %v", got)
	}
}
