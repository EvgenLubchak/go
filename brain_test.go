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

	// Контракт памʼяті тепер живе в Net, але глобаль лишилась ДЕФОЛТОМ для мереж,
	// створених без конфігу (як NewBrain нижче) — тож пінимо саме її.
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
			action := b.Step(state, false)
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
		q, _, _ := b.net.forwardQ(b.net.stackSteady(state)) // [ПАМ'ЯТЬ] проба усталеним стеком
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
				b.Step(state, seed%3 == 0)
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
	m := loadNetFrom(path, n.mem, n.gamma, n.clip) // той самий контракт, під який зберігали
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
	// Старий файл БЕЗ контракту (MemFrames == 0) мусить прийматись як є — інакше
	// рефакторинг знецінив би вже накопичені ваги.
	m2 := loadNetFrom(oldPath, n.mem, n.gamma, n.clip)
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
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true
	chaser, killer := ConfigLearner, ConfigKiller
	chaser.Count, killer.Count = 2, 2
	unitRoster = []UnitConfig{chaser, killer}

	units := newUnits()
	var killerNet, chaserNet *Net
	for i := range units {
		b := units[i].Brain
		if b == nil || b.net == nil {
			t.Fatalf("агент %d без мозку", i)
		}
		if units[i].Cfg.UsesFlowField {
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

	g := &Game{units: []Pixel{
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

	g := &Game{difficulty: 1.0, units: []Pixel{attacker, victim}}
	g.resolveImpacts()

	a, v := &g.units[0], &g.units[1]
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
		b.progress = 0 // нема власного руху → щільний член = 0
		b.dmgDealt, b.dmgTaken, b.kills = 2, 1, 1
		return b
	}
	plain, fighter := mk(false), mk(true)
	rPlain := plain.rewardFor(false, 0)
	rFight := fighter.rewardFor(false, 0)

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

// TestRosterConfigsAreComplete — [КОМАНДИ] страховка від класу помилок, який ми
// щойно зловили наживо: у конфігах забулись Faction і WeightsFile. Оскільки
// factionPlayer == 0, забутий Faction МОВЧКИ робить ворогів «своїми», а порожній
// WeightsFile злипає два типи в один вулик. Компілятор такого не бачить.
//
// [ТЕСТ НЕ ЗАЛЕЖИТЬ ВІД БАЛАНСУ] Count=0 — це не помилка, а свідомо вимкнений тип
// (так робимо ізольовані прогони). Перевіряємо УЗГОДЖЕНІСТЬ конфігів, а не те,
// який склад поля обрано зараз.
func TestRosterConfigsAreComplete(t *testing.T) {
	seenFile := map[string]string{}
	for _, cfg := range unitRoster {
		if cfg.Count < 0 {
			t.Errorf("відʼємний Count=%d", cfg.Count)
		}
		if cfg.Count == 0 {
			continue // тип свідомо вимкнено (ізольовані прогони) — не помилка
		}
		if cfg.Faction != factionEnemy && cfg.Faction != factionPlayer {
			t.Errorf("невідома фракція %d", cfg.Faction)
		}
		if !cfg.IsLearner {
			continue
		}
		if cfg.WeightsFile == "" {
			t.Errorf("тип-учень без WeightsFile → ділив би вулик з іншим типом")
			continue
		}
		// Один файл ваг = один тип мозку. Два різні типи з тим самим файлом
		// означали б, що вони вчаться в одну мережу з різними входами/нагородами.
		if prev, dup := seenFile[cfg.WeightsFile]; dup {
			t.Errorf("файл ваг %q ділять два типи (%s і цей) — вулики не розділені",
				cfg.WeightsFile, prev)
		}
		seenFile[cfg.WeightsFile] = cfg.WeightsFile
	}

	// На полі мають бути ОБИДВІ сторони — інакше командного бою не вийде.
	var enemies, allies int
	for _, cfg := range unitRoster {
		if cfg.Count == 0 {
			continue
		}
		if cfg.Faction == factionEnemy {
			enemies += cfg.Count
		} else {
			allies += cfg.Count
		}
	}
	t.Logf("склад поля: %d ворогів проти %d юнітів гравця", enemies, allies)
	if enemies == 0 || allies == 0 {
		t.Log("увага: одна зі сторін порожня — командного бою не буде")
	}
}

// TestFrozenPolicyStopsLearning перевіряє заморозку політики (клавіша L):
// навчання зупинене, ε=0. Без цього замір іде по РУХОМІЙ цілі — ваги повзуть
// прямо під час вимірювання, а 1% дій ще й випадкові.
//
// Тест навмисно перевіряє й ЗВОРОТНЕ (розморожені ваги таки змінюються) —
// інакше він проходив би і тоді, коли тренування зламане й не робить нічого.
func TestFrozenPolicyStopsLearning(t *testing.T) {
	savedRoster, savedFrozen, savedGRU := unitRoster, frozenPolicy, useGRU
	defer func() { unitRoster, frozenPolicy, useGRU = savedRoster, savedFrozen, savedGRU }()
	useGRU = false // перевіряємо стек-шлях: у нього детермінований поріг qMinReplay
	chaser := ConfigLearner
	chaser.Count = 1
	unitRoster = []UnitConfig{chaser}

	g := &Game{units: newUnits()}
	b := g.units[0].Brain
	if b == nil || b.net == nil {
		t.Fatal("юніт без мозку")
	}

	// Наповнюємо буфер, щоб train() мав на чому вчитись (інакше вийде з нього одразу).
	var s, s2 [brainInputs]float32
	for i := range s {
		s[i], s2[i] = 0.1*float32(i%3), 0.05*float32((i+1)%3)
	}
	for i := 0; i < qMinReplay*2; i++ {
		b.net.remember(transition{s: s, a: i % brainActions, r: 1, s2: s2})
	}

	frozenPolicy = true
	if eps := b.epsilon(); eps != 0 {
		t.Errorf("замороженій політиці потрібна ε=0, отримали %.4f", eps)
	}
	before, _, _ := b.net.forwardQ(s)
	for i := 0; i < 50; i++ {
		g.trainBrains()
	}
	after, _, _ := b.net.forwardQ(s)
	if before != after {
		t.Errorf("ваги зрушили при замороженій політиці: %v → %v", before, after)
	}

	frozenPolicy = false
	for i := 0; i < 50; i++ {
		g.trainBrains()
	}
	thawed, _, _ := b.net.forwardQ(s)
	if thawed == after {
		t.Error("після розморозки ваги не змінились — тренування не працює, тест був би пустим")
	}
}

// TestBodyRestShapeIsTheSquare закріплює головну обіцянку восьмикутного тіла:
// у СПОКОЇ (без деформації) вісім радіусів дають рівно той самий квадрат, що ми
// малювали раніше. Тобто перехід на полігон нічого не змінює візуально, поки
// мережа не почне його гнути — а отже ідентичність «пікселя» не втрачена.
//
// Перевіряємо буквально: вершина вздовж кожного напрямку мусить лежати НА контурі
// квадрата з півсторо́ною pixelSize/2, тобто max(|x|,|y|) == R.
func TestBodyRestShapeIsTheSquare(t *testing.T) {
	const R = pixelSize / 2
	for i := 0; i < brainActions; i++ {
		r := bodyRestRadius(i)
		x := dirs8[i][0] * r
		y := dirs8[i][1] * r
		m := float32(math.Max(math.Abs(float64(x)), math.Abs(float64(y))))
		if math.Abs(float64(m-R)) > 1e-4 {
			t.Errorf("напрямок %d: вершина (%.3f, %.3f) не на контурі квадрата: max=%.4f, треба %.4f",
				i, x, y, m, float32(R))
		}
	}
	// Діагональні радіуси мусять бути довшими за осьові саме в √2 разів.
	ratio := bodyRestRadius(1) / bodyRestRadius(0)
	if math.Abs(float64(ratio)-math.Sqrt2) > 1e-4 {
		t.Errorf("діагональ/вісь = %.5f, очікували √2 = %.5f", ratio, math.Sqrt2)
	}
}

// TestChargeHurtsHugDoesNot закріплює правило бою й ПОРЯДОК його обчислення.
//
// Механіка задумана так: «повільно зіштовхнулись = нічого; налетів = вкусив».
// Але гравець не бере участі в separation (його немає в boidMap), тож юніт міг
// стояти всередині нього, безперервно прискорюючись, і формально «налітати на
// повній» щокадру — смерть від обіймів за ~4 секунди.
//
// Лікування — pushOffPlayer, який гасить швидкість У бік гравця. Пастка в тому, що
// він МУСИТЬ викликатись ПІСЛЯ resolveImpacts: інакше на кадрі прильоту шкода
// рахувалася б по вже обнуленій швидкості й жоден удар не зараховувався б ніколи.
// Тест ловить обидва боки — і що кидок works, і що обійми ні.
func TestChargeHurtsHugDoesNot(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{} // без стін: перевіряємо саме зіткнення тіл
	defer func() { tileMap = savedMap }()

	newCase := func(velX float32) *Game {
		g := &Game{difficulty: 1.0}
		g.player = Pixel{X: 500, Y: 500, HP: 10, MaxHP: 10, Faction: factionPlayer}
		g.units = []Pixel{{
			X: 510, Y: 500, // перекриття 15px при pixelSize=25
			VelX: velX,
			HP:   2, MaxHP: 2,
			Faction: factionEnemy,
			Cfg:     ConfigLearner,
		}}
		return g
	}

	// Поріг удару для рою: max(MaxSpeed×impactSpeedFrac, impactMinSpeed).
	thr := impactThreshold(ConfigLearner.MaxSpeed)

	// 1) КИДОК: юніт праворуч від гравця летить УЛІВО на повній швидкості.
	g := newCase(-ConfigLearner.MaxSpeed)
	g.resolveImpacts()
	if g.player.HP != 9 {
		t.Errorf("кидок на швидкості %.2f (поріг %.2f) не завдав шкоди: HP %d, очікували 9",
			ConfigLearner.MaxSpeed, thr, g.player.HP)
	}

	// 2) ОБІЙМИ: той самий контакт, але швидкості зближення немає.
	g = newCase(0)
	g.resolveImpacts()
	if g.player.HP != 10 {
		t.Errorf("нерухомий контакт завдав шкоди: HP %d, очікували 10", g.player.HP)
	}

	// 3) pushOffPlayer розводить тіла й ВІДБИВАЄ юніта назовні.
	//    Саме відбивання, а не гасіння: із гасінням юніт застрягав на поверхні
	//    назавжди (за кадр мозок додає 0.3 при порозі 0.72) і удари зникали зовсім.
	g = newCase(-ConfigLearner.MaxSpeed)
	g.pushOffPlayer(&g.units[0])
	if g.units[0].VelX <= 0 {
		t.Errorf("юніт не відбився від гравця: VelX %.3f, очікували > 0", g.units[0].VelX)
	}
	if collides(g.player.X, g.player.Y, g.units[0].X, g.units[0].Y) {
		t.Errorf("юніт лишився всередині гравця: X %.1f проти %.1f", g.units[0].X, g.player.X)
	}

	// 4) Ключове: після відскоку юніт має ЗМОГУ знову набрати поріг. Перевіряємо
	//    напряму — розбігу треба лише кілька кадрів, і саме цього бракувало.
	frames := 0
	v := float32(0) // швидкість у бік гравця з нуля
	for v < thr && frames < 30 {
		v += brainForce
		if v > ConfigLearner.MaxSpeed {
			v = ConfigLearner.MaxSpeed
		}
		frames++
	}
	if v < thr {
		t.Errorf("юніт не може набрати поріг удару навіть за 30 кадрів: %.2f < %.2f", v, thr)
	}
}

// TestUnitsSeparateRegardlessOfFaction — фізика тіл діє МІЖ УСІМА юнітами, а шкода
// лише між ворожими. Це дві різні речі, і в resolveImpacts їх легко переплутати:
// раніше перевірка фракції стояла перед усім тілом циклу, тож додавання розштовхування
// туди ж мовчки лишило б своїх проникними одне для одного.
func TestUnitsSeparateRegardlessOfFaction(t *testing.T) {
	savedMap, savedFF := tileMap, friendlyFire
	tileMap = [boidMapH][boidMapW]bool{}
	friendlyFire = false
	defer func() { tileMap, friendlyFire = savedMap, savedFF }()

	pair := func(fa, fb int) *Game {
		g := &Game{difficulty: 1.0}
		g.player = Pixel{X: 50, Y: 50, HP: 10, MaxHP: 10, Faction: factionPlayer} // осторонь
		g.units = []Pixel{
			{X: 500, Y: 500, VelX: ConfigLearner.MaxSpeed, HP: 2, MaxHP: 2, Faction: fa, Cfg: ConfigLearner},
			{X: 510, Y: 500, HP: 2, MaxHP: 2, Faction: fb, Cfg: ConfigLearner},
		}
		return g
	}

	// Свої: шкоди немає, але тіла все одно розходяться.
	g := pair(factionEnemy, factionEnemy)
	g.resolveImpacts()
	if g.units[0].HP != 2 || g.units[1].HP != 2 {
		t.Errorf("свої завдали шкоди при friendlyFire=false: HP %d/%d", g.units[0].HP, g.units[1].HP)
	}
	if collides(g.units[0].X, g.units[0].Y, g.units[1].X, g.units[1].Y) {
		t.Error("свої лишились у перекритті — фізика не спрацювала")
	}

	// Чужі: і шкода, і розведення.
	g = pair(factionEnemy, factionPlayer)
	g.resolveImpacts()
	if g.units[1].HP != 1 {
		t.Errorf("кидок по ворогові не завдав шкоди: HP %d, очікували 1", g.units[1].HP)
	}
	if collides(g.units[0].X, g.units[0].Y, g.units[1].X, g.units[1].Y) {
		t.Error("вороги лишились у перекритті — фізика не спрацювала")
	}

	// Розходяться ОБИДВА (рівні маси), на відміну від випадку з гравцем.
	if g.units[0].X >= 500 {
		t.Errorf("нападник не відсунувся: X %.1f, очікували < 500", g.units[0].X)
	}
	if g.units[1].X <= 510 {
		t.Errorf("ціль не відсунулась: X %.1f, очікували > 510", g.units[1].X)
	}
}

// TestKnockbackOnlyOnLandedHit — віддача належить УДАРУ, а не дотику.
//
// Два різні механізми легко злити в один і отримати те, від чого йшли: bodyBounce
// пропорційний швидкості зіткнення (тиснуться на 0.3 — відскакують на 0.3, мікрорух),
// а knockback фіксований і великий, але лише коли шкода реально зарахувалась.
// Плюс перевіряємо, що імпульс ПЕРЕВИЩУЄ звичайну стелю швидкості: без піднятої
// стелі (knockSpeedMulti) кліп у updateUnits зʼїв би віддачу за перший же кадр.
func TestKnockbackOnlyOnLandedHit(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	setup := func(velX float32) *Game {
		g := &Game{difficulty: 1.0}
		g.player = Pixel{X: 50, Y: 50, HP: 10, MaxHP: 10, Faction: factionPlayer}
		g.units = []Pixel{
			{X: 500, Y: 500, VelX: velX, HP: 2, MaxHP: 2, Faction: factionEnemy, Cfg: ConfigLearner},
			{X: 510, Y: 500, HP: 2, MaxHP: 2, Faction: factionPlayer, Cfg: ConfigLearner},
		}
		return g
	}

	// Зарахований удар: ціль відлітає ШВИДШЕ за власну стелю, обом виставлено таймер.
	g := setup(ConfigLearner.MaxSpeed)
	g.resolveImpacts()
	victim, striker := g.units[1], g.units[0]
	if victim.VelX <= ConfigLearner.MaxSpeed {
		t.Errorf("віддача не перевищила стелю: VelX %.2f, MaxSpeed %.2f — кліп зʼїсть її за кадр",
			victim.VelX, ConfigLearner.MaxSpeed)
	}
	if victim.KnockTimer == 0 || striker.KnockTimer == 0 {
		t.Errorf("таймер відльоту не виставлено: ціль %d, нападник %d",
			victim.KnockTimer, striker.KnockTimer)
	}
	if striker.VelX >= 0 {
		t.Errorf("нападника не відсікло назад: VelX %.2f, очікували < 0", striker.VelX)
	}

	// Дотик без удару: жодної віддачі, лише дрібне розведення тіл.
	g = setup(0)
	g.resolveImpacts()
	if g.units[1].KnockTimer != 0 {
		t.Error("віддача спрацювала від простого дотику — це має робити лише удар")
	}
	if g.units[1].VelX > ConfigLearner.MaxSpeed {
		t.Errorf("дотик розігнав ціль понад стелю: VelX %.2f", g.units[1].VelX)
	}
}

// TestNarrowGapTolerance міряє, скільки позицій із 25 дозволяють пройти в прохід
// шириною в ОДИН тайл. Без вставки колайдера відповідь — рівно 1 піксель: тіло
// точно дорівнює дірці, і пройти можна лише потрапивши піксель-у-піксель. Саме це
// відчувалось як «застрягання на гострих кутах», хоч кути ні до чого.
//
// Тест числовий навмисно: «стало легше проходити» на око не перевіряється, а це
// число зникне при першій же правці геометрії, якщо його не пінити.
func TestNarrowGapTolerance(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	// Вільна лише колонка col; з боків — ТОВСТІ стіни на всю висоту.
	// Товсті навмисно: з однією колонкою стіни діапазон сканування виходив у
	// відкрите поле за нею, і воно зараховувалось як «прохід» (моя перша версія
	// тесту так і завищила допуск із 7 до 10).
	const col = 20
	for row := 0; row < boidMapH; row++ {
		for d := 1; d <= 3; d++ {
			tileMap[row][col-d] = true
			tileMap[row][col+d] = true
		}
	}

	y := float32(10 * pixelSize) // рядок усередині коридору
	pass := 0
	for x := (col - 1) * pixelSize; x < (col+2)*pixelSize; x++ {
		if !isWallRect(float32(x), y) {
			pass++
		}
	}

	want := 1 + 2*wallInset
	t.Logf("проходимих позицій: %d із %d (очікували ~%d при wallInset=%d)",
		pass, pixelSize, want, wallInset)

	if pass <= 1 {
		t.Errorf("допуск не зріс: %d позиція(ї) — тіло досі точно дорівнює дірці", pass)
	}
	if pass < want-1 || pass > want+1 {
		t.Errorf("допуск %d не відповідає wallInset=%d (очікували %d±1)", pass, wallInset, want)
	}
}

// TestBossIsLoneAndSeparate — бос мусить бути ОДИНАКОМ із власною мережею, і саме це
// робить його осмисленим типом. Виміряно, що вся перевага рою — спільний буфер
// досвіду; якщо бос випадково почне ділити мережу з роєм, він отримає той самий
// спільний досвід, і памʼять для нього знову перестане щось означати.
//
// Плюс перевіряємо UsesFlowField=false: із полем він завжди знає шлях до цілі, і
// згадувати нічого — вийшов би ще один вбивця з декоративною памʼяттю.
func TestBossIsLoneAndSeparate(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true // найгірший випадок: вулики ввімкнені, бос не має в них потрапити

	// Файли ваг ПІДМІНЯЄМО на неіснуючі: інакше newNetFor читав би реальні
	// brain_weights.json / boss_weights.json із робочої теки, і результат тесту
	// залежав би від того, що щойно записала гра. Нам тут важлива лише РІЗНІСТЬ
	// файлів, а не їхній вміст.
	swarm, boss := ConfigLearner, ConfigBoss
	swarm.Count = 3
	swarm.WeightsFile = "test_swarm_never_exists.json"
	bossFileForTest := "test_boss_never_exists.json"
	boss.WeightsFile = bossFileForTest
	// [GO: UnitConfig — ЗНАЧИМИЙ тип] Count правимо ДО запису в ростер: у слайс
	// потрапляє КОПІЯ, і правка після цього змінила б лише локальну змінну.
	boss.Count = 1 // на полі він мусить бути, хоч би що стояло в грі
	unitRoster = []UnitConfig{swarm, boss}

	// Count > 1, а не != 1: вимкнути тип нулем — законний спосіб ізолювати замір
	// (той самий принцип, що в TestRosterConfigsAreComplete). Перевіряємо задум
	// «бос не буває натовпом», а не конкретне балансне число.
	if ConfigBoss.Count > 1 {
		t.Errorf("боса %d штук — тип задуманий як ОДИНАК, на цьому тримається сенс памʼяті",
			ConfigBoss.Count)
	}
	if boss.UsesFlowField {
		t.Error("бос із flow-field завжди знає шлях → памʼяті нічого робити")
	}
	if ConfigBoss.WeightsFile == ConfigLearner.WeightsFile {
		t.Error("бос ділить файл ваг із роєм → отримає спільний досвід, і памʼять знову не важитиме")
	}

	units := newUnits()
	var bossNet *Net
	swarmNets := map[*Net]bool{}
	for i := range units {
		b := units[i].Brain
		if b == nil || b.net == nil {
			t.Fatalf("юніт %d без мозку", i)
		}
		if units[i].Cfg.WeightsFile == bossFileForTest {
			bossNet = b.net
		} else {
			swarmNets[b.net] = true
		}
	}
	if bossNet == nil {
		t.Fatal("боса немає на полі — не потрапив у ростер")
	}
	if swarmNets[bossNet] {
		t.Error("мережа боса збіглася з мережею рою — вулик його проглинув")
	}
}

// TestRestartKeepsBrains — рестарт скидає СВІТ, але не мозки.
//
// Пастка, від якої страхує: restart викликає побудову юнітів, а звичайний newUnits
// перечитує ваги з диска. Ваги ж пишуться лише на виході й на game over — тож без
// перенесення мереж кожне натискання R відкидало б усе навчання від початку сесії.
// Разом із вагами переноситься й БУФЕР ДОСВІДУ (він живе в Net), а він набирається
// тисячі кадрів.
func TestRestartKeepsBrains(t *testing.T) {
	savedRoster, savedShared, savedMap := unitRoster, sharedBrain, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	sharedBrain = true
	defer func() { unitRoster, sharedBrain, tileMap = savedRoster, savedShared, savedMap }()

	// Файл неіснуючий: щоб тест не читав реальні ваги з робочої теки.
	learner := ConfigLearner
	learner.Count = 2
	learner.WeightsFile = "test_restart_never_exists.json"
	unitRoster = []UnitConfig{learner}

	g := &Game{difficulty: 1.0, units: newUnits()}
	before := g.units[0].Brain.net
	if before == nil {
		t.Fatal("юніт без мережі")
	}

	// Кладемо в буфер мітку, щоб побачити, чи він виживе.
	var s, s2 [brainInputs]float32
	s[0] = 0.4242
	before.remember(transition{s: s, a: 1, r: 1, s2: s2})
	lenBefore := before.replayLen()

	g.restart()

	after := g.units[0].Brain.net
	if after != before {
		t.Error("після рестарту мережа інша — навчання сесії втрачено")
	}
	if after.replayLen() != lenBefore {
		t.Errorf("буфер досвіду скинувся: %d → %d", lenBefore, after.replayLen())
	}
	// Світ при цьому таки новий.
	if g.tick != 0 {
		t.Errorf("годинник не скинувся: tick %d", g.tick)
	}
	if g.paused {
		t.Error("рестарт із паузи лишив світ застиглим")
	}
}

// TestMemoryContractIsPerNetwork — головна обіцянка рефакторингу: два типи юнітів
// можуть мати РІЗНУ памʼять одночасно. Доти це було неможливо — Step і train
// дивились на глобальний useGRU, тобто «усе або ніщо».
//
// Заразом перевіряємо, що глобаль лишилась ДЕФОЛТОМ: тип, який нічого не вказав,
// мусить її успадкувати, інакше кожен конфіг був би змушений повторювати вибір.
func TestMemoryContractIsPerNetwork(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	savedGRU, savedFrames, savedSkip := useGRU, memFrames, stackSkip
	defer func() {
		unitRoster, sharedBrain = savedRoster, savedShared
		useGRU, memFrames, stackSkip = savedGRU, savedFrames, savedSkip
	}()
	sharedBrain = true
	useGRU, memFrames, stackSkip = false, 4, 10 // глобаль = стек 4/10

	// Файли неіснуючі: тест не має читати реальні ваги з робочої теки.
	mk := func(file string, kind MemoryKind, frames, skip int) UnitConfig {
		c := ConfigLearner
		c.Count, c.WeightsFile = 1, file
		c.Memory, c.MemFrames, c.StackSkip = kind, frames, skip
		return c
	}
	unitRoster = []UnitConfig{
		mk("test_mem_stack.json", MemoryStack, 1, 0),     // явно стек, без історії
		mk("test_mem_gru.json", MemoryGRU, 0, 0),         // явно GRU
		mk("test_mem_default.json", MemoryDefault, 0, 0), // успадковує глобаль
	}

	got := map[string]memContract{}
	for _, u := range newUnits() {
		if u.Brain == nil || u.Brain.net == nil {
			t.Fatal("юніт без мережі")
		}
		got[u.Cfg.WeightsFile] = u.Brain.net.mem
	}

	if c := got["test_mem_stack.json"]; c.gru || c.memFrames != 1 || c.stackSkip != 10 {
		t.Errorf("явний стек: отримали gru=%v frames=%d skip=%d", c.gru, c.memFrames, c.stackSkip)
	}
	if c := got["test_mem_gru.json"]; !c.gru {
		t.Error("явний GRU не застосувався — памʼять досі керується глобаллю")
	}
	if c := got["test_mem_default.json"]; c.gru || c.memFrames != 4 || c.stackSkip != 10 {
		t.Errorf("дефолт не успадкував глобаль: gru=%v frames=%d skip=%d", c.gru, c.memFrames, c.stackSkip)
	}

	// Найважливіше: стек і GRU СПІВІСНУЮТЬ на одному полі.
	if got["test_mem_stack.json"].gru == got["test_mem_gru.json"].gru {
		t.Error("два типи отримали однаковий шлях памʼяті — рефакторинг не працює")
	}
}

// TestLoadRejectsWrongMemoryContract — ваги навчені під конкретну форму входу, і
// завантажувати їх під іншу не можна.
//
// Це закриття СПРАВЖНЬОГО бага: раніше можна було мовчки завантажити GRU-навчену
// мережу на стек-шлях. Файл проходив валідацію (у ньому є обидва набори ваг), але
// W1/W2/W3 лежали на випадковій ініціалізації — GRU-навчання їх не торкалось. Ніякого
// попередження, просто випадкова політика. Ми робили саме це під час GRU-експериментів.
func TestLoadRejectsWrongMemoryContract(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/w.json"

	// gruSkip задаємо ЯВНО: контракт із нулем недійсний (resolveMemContract його
	// клампить до 1), а тут ми конструюємо вручну.
	trained := memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 1}
	n := NewNet()
	n.mem = trained
	if err := saveNetTo(n, path); err != nil {
		t.Fatal(err)
	}

	if loadNetFrom(path, trained, n.gamma, n.clip) == nil {
		t.Fatal("той самий контракт мусить вантажитись")
	}

	cases := []struct {
		name string
		want memContract
	}{
		{"інший ШЛЯХ памʼяті (GRU-ваги на стек)", memContract{gru: false, memFrames: 4, stackSkip: 10, gruSkip: 1}},
		{"інший КРОК семплів", memContract{gru: true, memFrames: 4, stackSkip: 60, gruSkip: 1}},
		{"інша ГЛИБИНА памʼяті", memContract{gru: true, memFrames: 1, stackSkip: 10, gruSkip: 1}},
		// Найважливіший випадок для цієї роботи: важіль BPTT змінює АЛГОРИТМ
		// (n-step із γ^gruSkip), тож ваги, навчені при кроці 1, не сумісні з 20.
		{"інший ВАЖІЛЬ BPTT", memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 20}},
	}
	for _, c := range cases {
		if loadNetFrom(path, c.want, n.gamma, n.clip) != nil {
			t.Errorf("%s: мережа завантажилась, хоч навчена під інший контракт", c.name)
		}
	}
}

// TestGruSkipRepeatsAndSumsReward — важіль BPTT цілиться в те, що градієнт не дістає
// далі 8 кадрів. Прорідження рішень розтягує ті самі 8 кроків на 8×gruSkip кадрів,
// але воно РОБОЧЕ лише разом із двома речами, і тест перевіряє саме їх:
//
//  1. між рішеннями дія ПОВТОРЮЄТЬСЯ (інакше h рухався б щокадру й нічого не змінилось);
//  2. нагороди за пропущені кадри СУМУЮТЬСЯ (інакше (N−1)/N сигналу зникає).
//
// Плюс метрики мусять лишитись покадровими — інакше вибірка зменшилась би в gruSkip
// разів і перестала бути порівнянною зі стек-шляхом.
func TestGruSkipRepeatsAndSumsReward(t *testing.T) {
	const skip = 5

	n := NewNet()
	n.mem = resolveMemContract(MemoryGRU, 0, 0, skip)
	b := NewBrainWith(n)

	var cur [baseInputs]float32
	cur[inDist] = 0.5
	cur[inVisible] = 0 // сліпий кадр → метрика мусить рахуватись

	first := b.stepGRU(cur, false)

	// Наступні skip-1 кадрів — та сама дія, без нового рішення.
	for i := 1; i < skip; i++ {
		b.progress = 1 // додатний прогрес: і нагорода, і blindClosed
		if got := b.stepGRU(cur, false); got != first {
			t.Fatalf("кадр %d: дія змінилась (%d → %d) — прорідження не діє", i, first, got)
		}
	}

	// Нагорода накопичилась за ВСІ пропущені кадри, а не за один.
	if b.gruAcc <= 0 {
		t.Fatalf("нагорода не накопичилась: gruAcc %.4f", b.gruAcc)
	}
	oneFrame := float32(rewardCloserScale) // progress=1 → саме стільки за кадр (без штрафів)
	if b.gruAcc < oneFrame*1.5 {
		t.Errorf("накопичено %.3f — схоже на ОДИН кадр (%.3f), а не на суму за %d",
			b.gruAcc, oneFrame, skip-1)
	}

	// Метрики покадрові: skip кадрів сліпих рішень, не одне.
	if b.mBlindN < skip-1 {
		t.Errorf("метрика порахувала %d сліпих кадрів замість ~%d — вибірка стала непорівнянною зі стеком",
			b.mBlindN, skip-1)
	}

	// На наступному кадрі — нове рішення, накопичувач скинуто.
	b.stepGRU(cur, false)
	if b.gruAcc != 0 || b.gruAccPow != 1 {
		t.Errorf("накопичувач не скинуто на новому рішенні: acc %.3f pow %.3f", b.gruAcc, b.gruAccPow)
	}
}

// TestGammaStepMatchesSkip — дискаунт мусить бути γ^gruSkip, бо крок відрізка накриває
// gruSkip кадрів. Із простим γ цінність майбутнього завищувалась би у стільки разів,
// скільки кадрів злито в один крок, — і n-step return перестав би бути коректним.
func TestGammaStepMatchesSkip(t *testing.T) {
	if got := resolveMemContract(MemoryGRU, 0, 0, 1).gammaStep(qGamma); got != qGamma {
		t.Errorf("крок 1 мусить давати рівно qGamma: %.5f проти %.5f", got, qGamma)
	}
	c := resolveMemContract(MemoryGRU, 0, 0, 20)
	want := float32(math.Pow(float64(qGamma), 20))
	if got := c.gammaStep(qGamma); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("γ^20: отримали %.6f, очікували %.6f", got, want)
	}
	// Здоровий сенс: за 20 кадрів при γ=0.95 лишається близько третини цінності.
	if g := c.gammaStep(qGamma); g < 0.30 || g > 0.40 {
		t.Errorf("γ^20 = %.3f — поза очікуваним діапазоном 0.30..0.40", g)
	}
}

// TestCombatOnlyRewardHasNoChaseTerm — головна ручка експерименту зі стражником.
//
// Наша щільна нагорода фактично ПІДКАЗУЄ напрямок: progressToward рахує прогрес від
// істинного напрямку навіть коли ціль не видно. Саме через це памʼять у нас нічого не
// дала. CombatOnly прибирає цей член — і тест пінить, що прибирає САМЕ його, а не
// заодно штрафи за стіни (вони походять від вусів, тобто від того, що агент бачить, і
// експерименту не псують).
func TestCombatOnlyRewardHasNoChaseTerm(t *testing.T) {
	mk := func(combatOnly bool) *Brain {
		b := NewBrain()
		b.combat, b.combatOnly = true, combatOnly
		b.progress = 1.0 // сильний «прогрес до цілі»
		return b
	}

	// 1) Щільний член: у звичайного є, у combatOnly немає.
	normal := mk(false).rewardFor(false, 0)
	only := mk(true).rewardFor(false, 0)
	if normal <= 0 {
		t.Fatalf("звичайна нагорода не побачила прогрес: %.4f", normal)
	}
	if only != 0 {
		t.Errorf("combatOnly отримав %.4f за прогрес — щільний член не прибрано", only)
	}

	// 2) Штрафи за стіни МУСЯТЬ лишитись: вони не підказують, де ціль.
	wall := mk(true)
	if r := wall.rewardFor(true, 0.5); r >= 0 {
		t.Errorf("combatOnly не отримав штрафу за стіну: %.4f", r)
	}

	// 3) Бойові члени працюють — інакше в нього не лишилось би сигналу зовсім.
	hit := mk(true)
	hit.dmgDealt = 1
	if r := hit.rewardFor(false, 0); r <= 0 {
		t.Errorf("combatOnly не отримав нагороди за влучання: %.4f — сигналу немає взагалі", r)
	}
	hurt := mk(true)
	hurt.dmgTaken = 1
	if r := hurt.rewardFor(false, 0); r >= 0 {
		t.Errorf("combatOnly не отримав штрафу за пропущений удар: %.4f", r)
	}
}

// TestPlayerMeleeCreditsBrain — удар пробілом мусить доходити до мозку цілі.
//
// Був справжній пропуск: playerAttack міняв HP напряму, обходячи applyImpactDamage,
// тож dmgTaken не реєструвався НІКОЛИ. Отже всі агенти з CombatReward учились не
// відчуваючи головної зброї гравця. Для стражника, у якого бойова нагорода — ЄДИНЕ
// джерело сигналу, це зробило б навчання беззмістовним.
func TestPlayerMeleeCreditsBrain(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { unitRoster, tileMap = savedRoster, savedMap }()

	w := ConfigWarden
	w.Count, w.WeightsFile = 1, "test_warden_never_exists.json"
	unitRoster = []UnitConfig{w}

	g := &Game{difficulty: 1.0, units: newUnits()}
	g.player = Pixel{X: g.units[0].X, Y: g.units[0].Y, HP: 10, MaxHP: 10, Faction: factionPlayer}

	b := g.units[0].Brain
	if b == nil {
		t.Fatal("стражник без мозку")
	}
	hpBefore := g.units[0].HP

	// Викликаємо тіло атаки напряму: сама playerAttack читає клавіатуру (ebiten),
	// якої в тесті немає. Перевіряємо саме нарахування, а не введення.
	g.applyPlayerMelee()

	if g.units[0].HP != hpBefore-attackDamage {
		t.Errorf("HP не зменшилось: %d → %d", hpBefore, g.units[0].HP)
	}
	if b.dmgTaken != attackDamage {
		t.Errorf("шкода не зарахована в мозок: dmgTaken %d, очікували %d", b.dmgTaken, attackDamage)
	}
}

// TestHorizonIsPerNetwork — γ і стеля цінності мусять бути властивістю МЕРЕЖІ.
//
// Причина стала очевидною на стражнику. При ЩІЛЬНІЙ нагороді горизонт майже не
// важить — сигнал є щокадру, і свіп γ 0.95/0.99/0.995 не дав різниці. При РОЗРІДЖЕНІЙ
// це вже не статистика, а арифметика: сигнал є лише в момент події, тож дискаунт
// напряму визначає, наскільки далеко назад дотягнеться кредит. γ=0.95 при 120 TPS —
// це 20 кадрів = 0.167с; подія через 120 кадрів дисконтується до 0.002, тобто
// невидима. Різні нагороди → різні горизонти, отже глобаллю це бути не може.
func TestHorizonIsPerNetwork(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true

	dense, sparse := ConfigLearner, ConfigWarden
	dense.Count, dense.WeightsFile = 1, "test_dense_never_exists.json"
	sparse.Count, sparse.WeightsFile = 1, "test_sparse_never_exists.json"
	sparse.Gamma = 0.99
	unitRoster = []UnitConfig{dense, sparse}

	got := map[string]*Net{}
	for _, u := range newUnits() {
		if u.Brain == nil || u.Brain.net == nil {
			t.Fatal("юніт без мережі")
		}
		got[u.Cfg.WeightsFile] = u.Brain.net
	}

	d, sp := got[dense.WeightsFile], got[sparse.WeightsFile]
	if d == nil || sp == nil {
		t.Fatal("не всі типи вийшли на поле")
	}

	if d.gamma != qGamma {
		t.Errorf("тип без Gamma не успадкував глобаль: %.4f проти %.4f", d.gamma, qGamma)
	}
	if math.Abs(float64(sp.gamma-0.99)) > 1e-6 {
		t.Errorf("явна Gamma не застосувалась: %.4f", sp.gamma)
	}
	if d.gamma == sp.gamma {
		t.Error("два типи отримали однаковий горизонт — рефакторинг не працює")
	}

	// СТЕЛЯ мусить масштабуватись разом із γ. Без цього довший горизонт уперся б у
	// кліп, і зміни не було б видно — з причини, яку створили ми самі. Ця помилка вже
	// раз коштувала нам вбивці (природна Q ≈ 14.8 при стелі 10).
	wantClip := qClip * (1 - qGamma) / (1 - 0.99)
	if math.Abs(float64(sp.clip-wantClip)) > 1e-3 {
		t.Errorf("стеля не масштабувалась: %.2f, очікували %.2f", sp.clip, wantClip)
	}
	if sp.clip <= d.clip {
		t.Errorf("довший горизонт мусить мати ВИЩУ стелю: %.2f проти %.2f", sp.clip, d.clip)
	}
}

// TestLoadRejectsWrongGamma — масштаб Q прямо залежить від γ (рівноважна ≈ r/(1−γ)),
// тож продовжувати навчання з іншим горизонтом означає мати не ту калібровку.
// Старі файли без записаної γ приймаються, інакше рефакторинг знецінив би ваги.
func TestLoadRejectsWrongGamma(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/w.json"

	n := NewNet()
	n.gamma, n.clip = 0.99, 125
	if err := saveNetTo(n, path); err != nil {
		t.Fatal(err)
	}

	if loadNetFrom(path, n.mem, 0.99, 125) == nil {
		t.Fatal("та сама γ мусить вантажитись")
	}
	if loadNetFrom(path, n.mem, qGamma, qClip) != nil {
		t.Error("ваги, навчені при γ=0.99, завантажились під γ=0.95 — калібровка Q не та")
	}
}
