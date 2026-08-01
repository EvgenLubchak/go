package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"
)

// TestQLearningTDUpdate перевіряє ядро Q-learning: після TD-оновлення оцінка
// Q(s, a) має зсунутись У БІК Беллман-цілі (reward + γ·max Q(s')).
// Запуск: go test -run TestQLearningTDUpdate -v
func TestQLearningTDUpdate(t *testing.T) {
	b := NewBrain()

	// Два довільні різні стани.
	var s, s2 [brainInputs]float32
	for i := range s {
		s[i] = 0.1 * float32(i%3)
		s2[i] = 0.05 * float32((i+1)%3)
	}
	const a = 3
	const reward = 1.0

	q2, _, _ := b.net.forwardQ(s2)
	target := clamp(reward+qGamma*q2[argmaxQ(q2)], -qClip, qClip)

	qBefore, _, _ := b.net.forwardQ(s)
	b.net.tdUpdate(s, a, reward, s2)
	qAfter, _, _ := b.net.forwardQ(s)

	distBefore := float32(math.Abs(float64(target - qBefore[a])))
	distAfter := float32(math.Abs(float64(target - qAfter[a]))) // має зменшитись
	t.Logf("Q(s,a): %.4f → %.4f  (target %.4f)", qBefore[a], qAfter[a], target)
	if distAfter >= distBefore {
		t.Errorf("Q(s,a) не наблизилось до цілі: було %.4f, стало %.4f", distBefore, distAfter)
	}
}

// TestQLearningChasesNoWalls перевіряє, що агент ВЧИТЬСЯ переслідувати:
// після тренування жадібна дія в середньому вказує в бік гравця (без стін).
// Запуск: go test -run TestQLearningChasesNoWalls -v
func TestQLearningChasesNoWalls(t *testing.T) {
	saved := localSight
	localSight = false // тест переслідування — з ПОВНОЮ спостережуваністю
	defer func() { localSight = saved }()

	savedGRU := useGRU
	useGRU = false // цей тест перевіряє СТЕК-шлях (Step/train/forwardQ)
	defer func() { useGRU = savedGRU }()

	// [ТЕСТ НЕ ЗАЛЕЖИТЬ ВІД ДИЗАЙНУ РІВНЯ] Прибираємо ВСІ стіни на час тесту.
	// Інакше правки levelLayout або pixelSize засівають «порожню» зону стінами →
	// whiskers ≠ 0 → reward губиться у штрафах за стіни, і агент вчиться їх
	// обходити, а не переслідувати. Тест зветься NoWalls — робимо це буквально.
	// [GO: масив — значимий тип] savedMap := tileMap копіює його повністю.
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	b := NewBrain()
	// Ворог у відкритій зоні без стін (whiskers ≈ 0).
	enemy := &Pixel{X: 1100, Y: 500, Cfg: ConfigLearner, Brain: b}
	player := &Pixel{X: 1300, Y: 600}
	// [ТЕСТ НЕ ЗАЛЕЖИТЬ ВІД ТЮНІНГУ] Фіксована швидкість, а НЕ ConfigLearner.MaxSpeed:
	// інакше кожна правка гейм-балансу зсуває збіжність тесту. Причина — швидкість
	// масштабує нагороду (reward = Δdist × rewardCloserScale), а більші TD-помилки
	// частіше впираються в кліп ±1 → сигнал тупіє. Тут перевіряємо САМ алгоритм.
	maxSpd := float32(0.8)

	dist := func() float32 {
		dx, dy := player.X-enemy.X, player.Y-enemy.Y
		return float32(math.Sqrt(float64(dx*dx + dy*dy)))
	}
	// Відкритий прямокутник без стін і подалі від країв (щоб whiskers=0).
	const x0, x1, y0, y1 = 950, 1450, 360, 740
	randOpen := func(lo, hi float32) float32 { return lo + rand.Float32()*(hi-lo) }

	// Епізодне тренування: щоразу нова геометрія, гравець нерухомий в епізоді.
	for ep := 0; ep < 400; ep++ {
		enemy.X, enemy.Y = randOpen(x0, x1), randOpen(y0, y1)
		player.X, player.Y = randOpen(x0, x1), randOpen(y0, y1)
		enemy.VelX, enemy.VelY = 0, 0
		b.hasPrev = false // новий епізод — забуваємо минулий перехід

		for step := 0; step < 150; step++ {
			state := GatherInputs(enemy, player)
			action := b.Step(state, dist(), false)
			b.net.train(qBatch) // Step лише збирає досвід; тренуємо явно (як g.trainBrains)
			enemy.VelX += dirs8[action][0] * brainForce
			enemy.VelY += dirs8[action][1] * brainForce
			spd := float32(math.Sqrt(float64(enemy.VelX*enemy.VelX + enemy.VelY*enemy.VelY)))
			if spd > maxSpd {
				enemy.VelX = enemy.VelX / spd * maxSpd
				enemy.VelY = enemy.VelY / spd * maxSpd
			}
			enemy.X = clamp(enemy.X+enemy.VelX, x0, x1)
			enemy.Y = clamp(enemy.Y+enemy.VelY, y0, y1)
		}
	}

	// Перевірка: для кількох напрямків до гравця жадібна дія має дивитись туди ж.
	enemy.X, enemy.Y = 1100, 500
	offsets := [][2]float32{{300, 0}, {-300, 0}, {0, 250}, {0, -250}, {220, 220}, {-220, -220}}
	var sumDot float32
	for _, off := range offsets {
		player.X, player.Y = enemy.X+off[0], enemy.Y+off[1]
		state := GatherInputs(enemy, player)
		q, _, _ := b.net.forwardQ(stackSteady(state)) // [ПАМ'ЯТЬ] проба усталеним стеком
		a := argmaxQ(q)
		n := float32(math.Sqrt(float64(off[0]*off[0] + off[1]*off[1])))
		sumDot += dirs8[a][0]*off[0]/n + dirs8[a][1]*off[1]/n
	}
	avgDot := sumDot / float32(len(offsets))
	t.Logf("середня узгодженість дії з напрямком до гравця: %.3f", avgDot)
	if avgDot < 0.3 {
		t.Errorf("агент не навчився переслідувати: avgDot=%.3f (очікували > 0.3)", avgDot)
	}
}

// TestSharedBrainNoRace імітує гру в режимі sharedBrain: багато горутин (як
// воркер-пул у calcAcceleration) одночасно викликають Step на агентах, що ДІЛЯТЬ
// одну мережу (forward-читання + remember під мютексом), потім ОДНОПОТОКОВО train.
// Має бути чисто під детектором гонок: go test -race -run TestSharedBrainNoRace
func TestSharedBrainNoRace(t *testing.T) {
	net := NewNet()
	const agents = 16
	brains := make([]*Brain, agents)
	for i := range brains {
		brains[i] = NewBrainWith(net) // усі ділять ОДНУ мережу
	}

	for frame := 0; frame < 200; frame++ {
		// Паралельна фаза: кожен агент у власній горутині (як воркер-пул).
		var wg sync.WaitGroup
		for i := range brains {
			wg.Add(1)
			go func(b *Brain, seed int) {
				defer wg.Done()
				var state [baseInputs]float32 // один кадр (Step склеїть у стек)
				state[0] = float32(seed%7) * 0.1
				state[5+seed%brainWhiskers] = 0.6
				b.Step(state, float32(50+seed), seed%3 == 0)
			}(brains[i], i)
		}
		wg.Wait()
		// Однопотокова фаза: тренуємо спільну мережу раз/кадр.
		net.train(qBatch)
	}
}

// TestGRULearnsSequence перевіряє напрям градієнтів BPTT: після навчання на
// відрізку з ПОЗИТИВНИМ reward оцінка Q обраних дій має ЗРОСТИ (рух до цілі),
// а не тікати. Якщо знак градієнта переплутано — Q падав би, і тест упав.
// Запуск: go test -run TestGRULearnsSequence -v
func TestGRULearnsSequence(t *testing.T) {
	old := useGRU
	useGRU = true
	defer func() { useGRU = old }()

	n := NewNet()

	// Фіксований відрізок (burn-in + навчальні) із чітким сигналом: reward=+1 для дії 2.
	var seq sequence
	for i := 0; i < seqTotal; i++ {
		for m := 0; m < baseInputs; m++ {
			seq.x[i][m] = float32(math.Sin(float64(i*7+m))) * 0.5
		}
		seq.a[i] = 2
		seq.r[i] = 1.0
	}
	for m := 0; m < baseInputs; m++ {
		seq.xEnd[m] = 0.1
	}

	// Сума Q(обраної дії) на НАВЧАЛЬНОМУ вікні (жива мережа, з прогрівом burn-in).
	qSum := func() float32 {
		var h [gruHidden]float32
		var s float32
		for i := 0; i < seqTotal; i++ {
			q, hn := n.forwardGRU(seq.x[i], h)
			h = hn
			if i >= seqBurnIn {
				s += q[seq.a[i]]
			}
		}
		return s
	}

	before := qSum()
	// < qTargetSync (1000), щоб target не синхронізувався в межах тесту.
	for i := 0; i < 500; i++ {
		n.tdUpdateSeq(seq)
	}
	after := qSum()

	if math.IsNaN(float64(after)) || math.IsInf(float64(after), 0) {
		t.Fatalf("Q став NaN/Inf після BPTT: %v", after)
	}
	if after <= before {
		t.Fatalf("GRU не вчиться: Q обраної дії не зросло (before=%.4f after=%.4f)", before, after)
	}
	t.Logf("Q(дію) до=%.4f після=%.4f — зросло, BPTT працює", before, after)
}

// TestSaveLoadGRURoundTrip перевіряє, що ваги GRU переживають save/load (round-trip),
// а старий файл без ваг GRU не ламає завантаження (GRU ініціалізується з нуля).
// Пишемо в тимчасовий файл, щоб не чіпати реальний brain_weights.json.
func TestSaveLoadGRURoundTrip(t *testing.T) {
	dir := t.TempDir()

	n := NewNet()
	n.Wz[0][0] = 0.4242 // характерні значення
	n.Wq[2][5] = -0.777
	n.W1[1][1] = 0.333 // і стек-вага

	path := dir + "/w.json"
	if err := saveNetTo(n, path); err != nil {
		t.Fatal(err)
	}
	m := loadNetFrom(path)
	if m == nil {
		t.Fatal("loadNetFrom повернув nil")
	}
	if m.Wz[0][0] != n.Wz[0][0] || m.Wq[2][5] != n.Wq[2][5] {
		t.Fatalf("GRU-ваги не round-trip: Wz=%v Wq=%v", m.Wz[0][0], m.Wq[2][5])
	}
	if m.W1[1][1] != n.W1[1][1] {
		t.Fatalf("стек-ваги не round-trip: %v", m.W1[1][1])
	}
	if m.tWz[0][0] != n.Wz[0][0] { // target має бути синхронізована на завантажене
		t.Fatalf("target GRU не синхронізовано: %v", m.tWz[0][0])
	}

	// Старий файл (HasGRU=false) → GRU ініціалізується (не нульовий), стек цілий.
	old := BrainData{
		Inputs: brainInputs, Hidden1: brainHidden1, Hidden2: brainHidden2, Actions: brainActions,
		W1: n.W1,
	}
	b, _ := json.MarshalIndent(old, "", "  ")
	oldPath := dir + "/old.json"
	if err := os.WriteFile(oldPath, b, 0644); err != nil {
		t.Fatal(err)
	}
	m2 := loadNetFrom(oldPath)
	if m2 == nil {
		t.Fatal("старий файл: loadNetFrom повернув nil")
	}
	var nonZero bool
	for i := 0; i < gruHidden && !nonZero; i++ {
		for j := 0; j < baseInputs; j++ {
			if m2.Wz[i][j] != 0 {
				nonZero = true
				break
			}
		}
	}
	if !nonZero {
		t.Fatal("старий файл: GRU-ваги лишились нульовими (initGRU не спрацював)")
	}
}

// TestKillerHasSeparateHive — [ВБИВЦЯ] перевіряє головне у кроці 2b-1: вбивці й
// рій вчаться в РІЗНИХ мережах і зберігаються в різні файли. Якщо це зламається,
// зміна reward для вбивці мовчки перетре тонко налаштований рій.
func TestKillerHasSeparateHive(t *testing.T) {
	// [СКЛАД ПОЛЯ] Тест не залежить від бойового балансу: підставляємо власний
	// roster (по одному юніту кожного типу), а справжній повертаємо через defer.
	savedRoster, savedShared := enemyRoster, sharedBrain
	defer func() { enemyRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true
	chaser, killer := ConfigLearner, ConfigKiller
	chaser.Count, killer.Count = 2, 2
	enemyRoster = []EnemyConfig{chaser, killer}

	enemies := newEnemies()
	var killerNet, chaserNet *Net
	for i := range enemies {
		b := enemies[i].Brain
		if b == nil || b.net == nil {
			t.Fatalf("агент %d без мозку", i)
		}
		if enemies[i].Cfg.UsesFlowField {
			if killerNet == nil {
				killerNet = b.net
			} else if b.net != killerNet {
				t.Fatal("вбивці не ділять один вулик")
			}
		} else {
			if chaserNet == nil {
				chaserNet = b.net
			} else if b.net != chaserNet {
				t.Fatal("переслідувачі не ділять один вулик")
			}
		}
	}
	if killerNet == nil || chaserNet == nil {
		t.Fatal("створено не обидва типи ворогів")
	}
	if killerNet == chaserNet {
		t.Fatal("вбивця й рій ділять ОДНУ мережу — типи не розділені")
	}
	if killerNet.file != killerFile {
		t.Fatalf("мережа вбивці пише в %q, а мала в %q", killerNet.file, killerFile)
	}
	if chaserNet.file != brainFile {
		t.Fatalf("мережа рою пише в %q, а мала в %q", chaserNet.file, brainFile)
	}
}

// TestSaveBrainsWritesEachHive — saveBrains зберігає КОЖНУ унікальну мережу у її
// власний файл, а ефемерні (file == "") пропускає. Пишемо у tmp, не чіпаючи
// справжні ваги.
func TestSaveBrainsWritesEachHive(t *testing.T) {
	dir := t.TempDir()
	hiveA, hiveB, ephemeral := NewNet(), NewNet(), NewNet()
	hiveA.file = dir + "/a.json"
	hiveB.file = dir + "/b.json"
	// ephemeral.file лишається "" → не зберігається (як мозок-жертва в self-play)

	g := &Game{enemies: []Pixel{
		{Brain: NewBrainWith(hiveA)}, {Brain: NewBrainWith(hiveA)},
		{Brain: NewBrainWith(hiveB)},
		{Brain: NewBrainWith(ephemeral)},
	}}
	g.saveBrains()

	for _, p := range []string{hiveA.file, hiveB.file} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("не збережено %s: %v", p, err)
		}
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 2 {
		t.Fatalf("у теці %d файлів, очікували 2 (ефемерна мережа не мала зберігатись)", len(entries))
	}
}

// TestCombatRewardAndAttribution — [БІЙ] крок 2b-3: перевіряє два звʼязані місця.
//  1. resolveImpacts правильно АТРИБУТУЄ удар: хто завдав, хто отримав, хто добив;
//  2. rewardFor перетворює це на нагороду — але ЛИШЕ для мозків із combat=true,
//     а лічильники обнуляє ЗАВЖДИ (інакше в рою вони росли б вічно).
func TestCombatRewardAndAttribution(t *testing.T) {
	// --- Частина 1: атрибуція в resolveImpacts ---
	attacker := Pixel{X: 100, Y: 100, VelX: ConfigKiller.MaxSpeed, Cfg: ConfigKiller,
		HP: 3, MaxHP: 3, Faction: factionEnemy, Brain: NewBrain()}
	victim := Pixel{X: 110, Y: 100, Cfg: ConfigLearner, // стоїть на місці
		HP: 1, MaxHP: 2, Faction: factionPlayer, Brain: NewBrain()}

	g := &Game{difficulty: 1.0, enemies: []Pixel{attacker, victim}}
	g.resolveImpacts()

	a, v := &g.enemies[0], &g.enemies[1]
	if a.Brain.dmgDealt != impactDamage {
		t.Fatalf("нападнику не зараховано шкоду: dmgDealt=%d", a.Brain.dmgDealt)
	}
	if v.Brain.dmgTaken != impactDamage {
		t.Fatalf("жертві не зараховано отриману шкоду: dmgTaken=%d", v.Brain.dmgTaken)
	}
	if a.Brain.kills != 1 {
		t.Fatalf("ціль загинула (HP=%d), але вбивство не зараховано: kills=%d", v.HP, a.Brain.kills)
	}
	if v.Brain.dmgDealt != 0 {
		t.Fatalf("нерухома жертва не мала завдати шкоди, а має dmgDealt=%d", v.Brain.dmgDealt)
	}

	// --- Частина 2: нагорода з цих лічильників ---
	// Той самий стан, але два мозки: бойовий і звичайний. Різниця має дорівнювати
	// рівно бойовим членам.
	mk := func(combat bool) *Brain {
		b := NewBrain()
		b.combat = combat
		b.prevDist = 100 // дистанція не змінилась → щільний член = 0
		b.dmgDealt, b.dmgTaken, b.kills = 2, 1, 1
		return b
	}
	plain, fighter := mk(false), mk(true)
	rPlain := plain.rewardFor(100, false, 0)
	rFight := fighter.rewardFor(100, false, 0)

	want := float32(rewardDamageDealt*2 + rewardDamageTaken*1 + rewardKill*1)
	if got := rFight - rPlain; got != want {
		t.Fatalf("бойова добавка %.2f, очікували %.2f", got, want)
	}
	if rPlain != 0 {
		t.Fatalf("небойовий мозок мав отримати 0, а отримав %.2f", rPlain)
	}
	// Лічильники обнуляються в ОБОХ.
	for name, b := range map[string]*Brain{"небойовий": plain, "бойовий": fighter} {
		if b.dmgDealt != 0 || b.dmgTaken != 0 || b.kills != 0 {
			t.Fatalf("%s мозок не обнулив лічильники: %d/%d/%d",
				name, b.dmgDealt, b.dmgTaken, b.kills)
		}
	}
	t.Logf("бойова добавка = %.1f (шкода +%.0f/−%.0f, вбивство +%.0f)",
		want, rewardDamageDealt, -rewardDamageTaken, rewardKill)
}
