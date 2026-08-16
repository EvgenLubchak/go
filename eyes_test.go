package main

import (
	"math"
	"testing"
)

// TestEyesFollowPerception — зіниця веде за СПРИЙНЯТТЯМ, а не малює життя.
//
// Це головна вимога до обличчя, і вона не косметична. Тіло вже показує НАМІР (форма з
// Q), тож якби очі показували те саме, ми отримали б другу копію того самого приладу.
// Очі взяли інший канал: куди агент дивиться і чи бачить ціль узагалі.
//
// Побічно це вперше робить видимим localSight: сховався за стіною — і видно, як тебе
// загубили. Тому «розфокус при втраті» перевіряється тут нарівні з наведенням.
func TestEyesFollowPerception(t *testing.T) {
	u := &Pixel{X: 100, Y: 100, Brain: NewBrain()}

	// --- бачить ціль праворуч → зіниця йде праворуч
	u.Brain.sees = true
	u.Brain.gazeX, u.Brain.gazeY = 1, 0
	for i := 0; i < 200; i++ {
		updateFace(u)
	}
	if u.Pupil[0] < 0.9 || math.Abs(float64(u.Pupil[1])) > 0.05 {
		t.Errorf("зіниця не навелась на ціль праворуч: %v", u.Pupil)
	}

	// --- ціль ЗНИКЛА → зіниця мусить повернутись у центр
	u.Brain.sees = false
	before := u.Pupil[0]
	updateFace(u)
	if u.Pupil[0] >= before {
		t.Error("після втрати цілі зіниця не рушила до центра")
	}
	for i := 0; i < 200; i++ {
		updateFace(u)
	}
	if math.Abs(float64(u.Pupil[0])) > 0.05 || math.Abs(float64(u.Pupil[1])) > 0.05 {
		t.Errorf("зіниця не розфокусувалась: %v", u.Pupil)
	}

	// --- ВІДСТАВАННЯ, а не клацання. Якби зіниця стрибала за один кадр, втрата цілі
	// читалась би як збій, а не як «загубив», і вся ідея розсипалась би.
	u.Brain.sees = true
	u.Brain.gazeX, u.Brain.gazeY = 0, 1
	updateFace(u)
	if u.Pupil[1] > 0.9 {
		t.Errorf("зіниця стрибнула на ціль за один кадр (%v) — це вимикач, а не погляд",
			u.Pupil[1])
	}
}

// TestPlayerEyesFollowMovement — у гравця мозку немає, тож очі дивляться в бік
// ВЛАСНОГО РУХУ. Це єдина чесна відповідь на «куди він спрямований»: нічого не
// вигадуємо й нічого не імітуємо.
func TestPlayerEyesFollowMovement(t *testing.T) {
	p := newPlayer()
	p.VelX, p.VelY = 0, -4 // летить угору
	for i := 0; i < 200; i++ {
		updateFace(&p)
	}
	if p.Pupil[1] > -0.9 {
		t.Errorf("зіниці гравця не дивляться в бік руху: %v", p.Pupil)
	}

	// Зупинився — погляд у центр. Поріг швидкості потрібен, бо біля нуля напрямок
	// вироджений: дрібне тремтіння дало б зіницям смикатись без причини.
	p.VelX, p.VelY = 0, 0
	for i := 0; i < 200; i++ {
		updateFace(&p)
	}
	if math.Abs(float64(p.Pupil[1])) > 0.05 {
		t.Errorf("на зупинці погляд не повернувся в центр: %v", p.Pupil)
	}
}

// TestEyesRideTheBodySpring — обличчя їде за пружиною тіла, як ворс і кінцівки.
// Без цього стиснутий юніт носив би завелике обличчя, і воно читалось би як наклеєне.
func TestEyesRideTheBodySpring(t *testing.T) {
	const cx, cy = 100, 100
	lx1, _ := eyeRoot(0, cx, cy, 1.0)
	rx1, _ := eyeRoot(1, cx, cy, 1.0)
	lx2, ly2 := eyeRoot(0, cx, cy, 0.5)

	if rx1 <= lx1 {
		t.Error("праве око не праворуч від лівого")
	}
	if got, want := cx-lx2, (cx-lx1)/2; math.Abs(float64(got-want)) > 1e-4 {
		t.Errorf("очі не звузились удвічі при масштабі 0.5: %v проти %v", got, want)
	}
	if ly2 >= cy {
		t.Error("очі мусять бути ВИЩЕ центра тіла — нижче лишається місце для рота")
	}
}

// TestGazeIsWiredToVisibility — поле sees справді приходить із POMDP-видимості.
//
// Цю перевірку додано після мутації: попередні тести задавали sees напряму, тож
// підміна «sees = true завжди» лишала їх зеленими. Тобто вони перевіряли ФІЗИКУ
// зіниці, але не те, ЩО нею керує, — а без цього шва обличчя не показує нічого.
//
// Саме тут і живе вся цінність очей: localSight був невидимий, і тільки цей звʼязок
// робить його видимим.
func TestGazeIsWiredToVisibility(t *testing.T) {
	savedMap, savedSight := tileMap, localSight
	tileMap, localSight = [boidMapH][boidMapW]bool{}, true // чисте поле, POMDP увімкнено
	defer func() { tileMap, localSight = savedMap, savedSight }()

	enemy := &Pixel{X: 400, Y: 400, Cfg: ConfigWarden, Brain: NewBrain()}

	near := &Pixel{X: 400 + sightRange/2, Y: 400}
	GatherInputs(enemy, near)
	if !enemy.Brain.sees {
		t.Error("ціль у межах зору й по прямій — агент мусить її бачити")
	}
	if enemy.Brain.gazeX <= 0.9 {
		t.Errorf("погляд не вказує на ціль праворуч: gazeX = %v", enemy.Brain.gazeX)
	}

	far := &Pixel{X: 400 + sightRange*2, Y: 400}
	GatherInputs(enemy, far)
	if enemy.Brain.sees {
		t.Errorf("ціль за %0.f px при sightRange = %0.f — бачити її агент не може",
			sightRange*2, sightRange)
	}
}

// TestMouthShowsVulnerability — рот відкривається саме тоді, коли ціль безкарна.
//
// Це ІГРОВА інформація, а не діагностика: форма тіла вже показує намір мережі, і
// другий прилад на тому ж юніті лише зашумив би. Рот відповідає на питання, за яким
// можна діяти: «бити зараз чи він ухилиться?»
//
// Джерела різні, бо ривок є лише в гравця, а ухилення — лише в юнітів. Спільне одне:
// у цей проміжок захиститись неможливо.
func TestMouthShowsVulnerability(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*Pixel)
		want float32
	}{
		{"гравець у відході після ривка", func(p *Pixel) { p.DashPhase = dashPhaseRecovery }, 1},
		{"гравець у замаху — ще зібраний", func(p *Pixel) { p.DashPhase = dashPhaseWindup }, 0},
		{"юніт у відході після кидка", func(p *Pixel) { p.DodgeRecover = dodgeRecovery }, 1},
		{"юніт лише на перезарядці", func(p *Pixel) { p.DodgeCooldown = dodgeCooldown }, 0},
		{"юніт готовий", func(p *Pixel) {}, 0},
	} {
		p := &Pixel{}
		tc.set(p)
		if got := mouthTargetOf(p); got != tc.want {
			t.Errorf("%s: ціль рота %v, очікували %v", tc.name, got, tc.want)
		}
	}

	// ВІДСТАВАННЯ, а не клацання: різкий перехід читався б як блимання індикатора,
	// плавний — як зміна стану істоти. Той самий механізм, що в зіницях і ворсі.
	p := &Pixel{DodgeRecover: dodgeRecovery}
	updateFace(p)
	if p.Mouth >= 1 {
		t.Errorf("рот розкрився за один кадр (%v) — це індикатор, а не обличчя", p.Mouth)
	}
	for i := 0; i < 300; i++ {
		updateFace(p)
	}
	if p.Mouth < 0.95 {
		t.Errorf("рот не дійшов до розкритого: %v", p.Mouth)
	}
	p.DodgeRecover = 0
	for i := 0; i < 300; i++ {
		updateFace(p)
	}
	if p.Mouth > 0.05 {
		t.Errorf("рот не закрився після перезарядки: %v", p.Mouth)
	}
}

// TestMouthChangesShapeNotJustSize — головна властивість рота: при розкритті
// змінюється ФОРМА, а не лише розмір.
//
// Різниця не косметична. Ширша щілина — це градація, яку треба порівнювати очима;
// кругле «о» — інша фігура, і вона читається однозначно з першого погляду. Для
// приладу, за яким гравець вирішує «бити зараз чи він ухилиться», однозначність
// важливіша за плавність.
func TestMouthChangesShapeNotJustSize(t *testing.T) {
	wc, hc, rc := mouthShape(0, 1) // риска
	wo, ho, ro := mouthShape(1, 1) // «о»

	// Перевіряємо СПІВВІДНОШЕННЯ, а не конкретні числа. Перша версія цього тесту
	// вимагала рівно кола (wo == ho) — і це була моя ДИЗАЙНЕРСЬКА ПРЕФЕРЕНЦІЯ, вбита в
	// перевірку як інваріант. Щойно Євген підкрутив рот під овал, тест почав валити
	// цілком правильний код. Тест мусить стерегти ВЛАСТИВІСТЬ, а не смак.
	//
	// Справжня властивість одна: форма мусить ПОМІТНО змінитись, бо на цьому й тримається
	// читабельність приладу. Ширша щілина — градація, яку треба порівнювати; інша фігура
	// впізнається з першого погляду.
	if wc <= hc*2 {
		t.Errorf("закритий рот не читається як РИСКА: %.1f × %.1f", wc, hc)
	}
	if wo >= wc {
		t.Error("рот при розкритті мусить ЗВУЖУВАТИСЬ — інакше це пігулка, а не «о»")
	}
	if ho <= hc {
		t.Error("рот при розкритті мусить рости у висоту")
	}
	// Співвідношення сторін мусить впасти щонайменше вдвічі: саме це й перетворює
	// риску на округлу фігуру, хай навіть овальну, а не ідеально круглу.
	if arC, arO := wc/hc, wo/ho; arO > arC/2 {
		t.Errorf("форма змінилась замало: %.1f:1 → %.1f:1 — читатиметься як та сама щілина",
			arC, arO)
	}
	_ = rc

	// Кінці ніколи не товщі за саму фігуру: інакше кола вилізли б за капсулу й
	// силует розповз би ся замість того, щоб зійтись у коло.
	for _, open := range []float32{0, 0.25, 0.5, 0.75, 1} {
		w, _, r := mouthShape(open, 1)
		if r > w/2+1e-4 {
			t.Errorf("при %.2f радіус кінця %.2f більший за півширини %.2f", open, r, w/2)
		}
	}
	// Радіус кінця — рівно піввисоти, доки він у неї вміщується. Це й робить фігуру
	// капсулою, а не прямокутником із приклеєними колами.
	if want := ho / 2; math.Abs(float64(ro-want)) > 1e-4 && ro != wo/2 {
		t.Errorf("радіус кінця %.2f не дорівнює ні піввисоти (%.2f), ні півширини (%.2f)",
			ro, want, wo/2)
	}

	// Пружина тіла масштабує все обличчя, як ворс і кінцівки.
	w1, h1, _ := mouthShape(0.5, 1)
	w2, h2, _ := mouthShape(0.5, 0.5)
	if math.Abs(float64(w2-w1/2)) > 1e-4 || math.Abs(float64(h2-h1/2)) > 1e-4 {
		t.Error("рот не їде за пружиною тіла")
	}
}
