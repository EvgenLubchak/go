package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
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
	b.net.tdUpdate(s, a, reward, s2, false)
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
			// [УХИЛЕННЯ] Девʼята дія без напрямку — юніт просто не прискорюється,
			// як і в calcAcceleration. Без цієї перевірки тест падає на dirs8[8].
			if action < brainWhiskers {
				enemy.VelX += dirs8[action][0] * brainForce
				enemy.VelY += dirs8[action][1] * brainForce
			}
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
		if a >= brainWhiskers {
			continue // ухилення напрямку не має, у скалярний добуток не входить
		}
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
	// НАПРЯМКІВ вісім, а ДІЙ девʼять — після появи ухилення це різні числа, і геометрія
	// рахується по перших. Раніше тут стояло brainActions і працювало лише тому, що
	// dirs8 був оголошений на brainActions із мовчазним нулем у девʼятому елементі.
	for i := 0; i < brainWhiskers; i++ {
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

	// Кладемо щось у лічильники заміру, інакше перевірка їхнього скидання пуста.
	g.units[0].Brain.mBlindN = 5
	g.metrics.collect(g)
	if g.metrics.unitFrames == 0 {
		t.Fatal("лічильники не наповнились — перевірка скидання була б пустою")
	}

	g.restart()

	after := g.units[0].Brain.net
	if after != before {
		t.Error("після рестарту мережа інша — навчання сесії втрачено")
	}
	if after.replayLen() != lenBefore {
		t.Errorf("буфер досвіду скинувся: %d → %d", lenBefore, after.replayLen())
	}
	// [МЕТРИКИ] Лічильники заміру описують світ, тож на рестарті мусять скинутись.
	// Без цього m.agents накопичував Brain-и через усі життя, і «ag» на панелі
	// показував кількість смертей, а не юнітів.
	if g.metrics.unitFrames != 0 || g.metrics.agents != nil {
		t.Errorf("лічильники заміру не скинуто: unitFrames %d, agents %d",
			g.metrics.unitFrames, len(g.metrics.agents))
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

	// gruSkip, actSkip і nStep задаємо ЯВНО: контракт із нулем недійсний
	// (resolveMemContract клампить їх до 1), а тут ми конструюємо вручну.
	trained := memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 1, actSkip: 1, nStep: 1}
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
		{"інший ШЛЯХ памʼяті (GRU-ваги на стек)", memContract{gru: false, memFrames: 4, stackSkip: 10, gruSkip: 1, actSkip: 1, nStep: 1}},
		{"інший КРОК семплів", memContract{gru: true, memFrames: 4, stackSkip: 60, gruSkip: 1, actSkip: 1, nStep: 1}},
		{"інша ГЛИБИНА памʼяті", memContract{gru: true, memFrames: 1, stackSkip: 10, gruSkip: 1, actSkip: 1, nStep: 1}},
		// Найважливіший випадок для цієї роботи: важіль BPTT змінює АЛГОРИТМ
		// (n-step із γ^gruSkip), тож ваги, навчені при кроці 1, не сумісні з 20.
		{"інший ВАЖІЛЬ BPTT", memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 20, actSkip: 1, nStep: 1}},
		// Те саме для повтору дії: він теж міняє АЛГОРИТМ (бутстрап через γ^actSkip),
		// тож ваги, навчені при кроці 1, не сумісні з 15.
		{"інший ПОВТОР ДІЇ", memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 1, actSkip: 15, nStep: 1}},
		// n-step теж міняє АЛГОРИТМ (ціль = сума за n кроків, бутстрап на γ^n).
		{"інший n-STEP", memContract{gru: true, memFrames: 4, stackSkip: 10, gruSkip: 1, actSkip: 1, nStep: 5}},
	}
	for _, c := range cases {
		if loadNetFrom(path, c.want, n.gamma, n.clip) != nil {
			t.Errorf("%s: мережа завантажилась, хоч навчена під інший контракт", c.name)
		}
	}
}

// TestActSkipRepeatsAndSumsReward — повтор дії для СТЕК-шляху.
//
// Навіщо: стражник мусить ухилятись, а це 45 кадрів ПОСЛІДОВНОЇ дії в один бік. При
// рішенні щокадру це 45 рішень поспіль, кожне з яких може зірватись на ε-шумі чи
// дрібній різниці Q. При actSkip = 15 їх ТРИ. Плюс кредит доходить у 15 разів швидше:
// одношаговий TD переносить цінність на один КРОК за оновлення, а крок тепер накриває
// 15 кадрів.
//
// Механізм робочий лише разом із трьома речами, і тест перевіряє саме їх — кожна
// ламається молча:
//
//  1. між рішеннями дія ПОВТОРЮЄТЬСЯ;
//  2. нагороди за пропущені кадри СУМУЮТЬСЯ з дискаунтом (інакше (N−1)/N зникає);
//  3. зсув історії спостережень іде СВОЇМ темпом (stackSkip), а не темпом рішень —
//     інакше вікно памʼяті мовчки розтяглось би в actSkip разів.
func TestActSkipRepeatsAndSumsReward(t *testing.T) {
	const skip = 5

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 4, 2, 0, skip)
	b := NewBrainWith(n)
	b.combat = true

	var in [baseInputs]float32
	in[inVisible] = 1

	// Прогріваємо один крок, щоб зʼявився hasPrev.
	first := b.stepStack(in, false)
	actions := []int{first}
	// Даємо шкоду ЩОКАДРУ: якщо сумування зламане, у переході осяде лише частина.
	for i := 0; i < skip; i++ {
		b.dmgDealt = 1
		actions = append(actions, b.stepStack(in, false))
	}

	// 1) Дія повторювалась усі skip кадрів після рішення.
	for i := 1; i < skip; i++ {
		if actions[i] != actions[0] {
			t.Fatalf("кадр %d: дія змінилась %d → %d, хоч рішення раз на %d кадрів",
				i, actions[0], actions[i], skip)
		}
	}

	// 2) Нагорода підсумована з дискаунтом. Очікуємо Σ γ^k · r за skip кадрів.
	if b.net.replayLen() == 0 {
		t.Fatal("жодного переходу в буфері")
	}
	var want float32
	pow := float32(1)
	for i := 0; i < skip; i++ {
		want += pow * rewardDamageDealt
		pow *= n.gamma
	}
	got := b.net.replay[b.net.replayLen()-1].r
	if diff := got - want; diff > 0.01 || diff < -0.01 {
		t.Errorf("нагорода кроку %.4f, чекали Σγ^k·r = %.4f — сумування зламане",
			got, want)
	}

	// 3) Вікно памʼяті не поїхало: за skip+1 кадрів при stackSkip=2 мусило статись
	//    (skip+1)/2 зсувів, тобто зсув живе своїм темпом, а не темпом рішень.
	b2 := NewBrainWith(n)
	shifts := 0
	prev := b2.frames
	for i := 0; i < 12; i++ {
		in[0] = float32(i) // унікальний кадр, щоб зсув було видно
		b2.stepStack(in, false)
		if b2.frames != prev {
			shifts++
			prev = b2.frames
		}
	}
	if shifts < 5 {
		t.Errorf("за 12 кадрів при stackSkip=2 сталось лише %d зсувів історії — "+
			"зсув привʼязали до РІШЕНЬ замість кадрів, і вікно памʼяті розтяглось", shifts)
	}
}

// TestActSkipBootstrapsPerStep — бутстрап мусить іти через γ^actSkip, а не через γ.
//
// Крок тепер накриває actSkip кадрів, і дисконтувати його як один кадр означало б
// рахувати майбутнє дорожчим, ніж воно є. Та сама помилка, що ми вже ловили в
// gruSkip — тому й перевіряємо тим самим способом.
func TestActSkipBootstrapsPerStep(t *testing.T) {
	c := resolveMemContract(MemoryStack, 4, 10, 0, 15)
	want := float32(math.Pow(float64(qGamma), 15))
	if got := c.gammaStep(qGamma); got < want*0.999 || got > want*1.001 {
		t.Errorf("gammaStep = %.6f, чекали γ^15 = %.6f", got, want)
	}
	if c.decisionSkip() != 15 {
		t.Errorf("decisionSkip = %d, чекали 15", c.decisionSkip())
	}
	// GRU-шлях мусить читати СВІЙ важіль, а не стековий.
	g := resolveMemContract(MemoryGRU, 4, 10, 7, 15)
	if g.decisionSkip() != 7 {
		t.Errorf("GRU decisionSkip = %d, чекали gruSkip 7", g.decisionSkip())
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
	n.mem = resolveMemContract(MemoryGRU, 0, 0, skip, 0)
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
	if got := resolveMemContract(MemoryGRU, 0, 0, 1, 0).gammaStep(qGamma); got != qGamma {
		t.Errorf("крок 1 мусить давати рівно qGamma: %.5f проти %.5f", got, qGamma)
	}
	c := resolveMemContract(MemoryGRU, 0, 0, 20, 0)
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

// wardenTestGame — стражник і гравець в одній точці, без стін. Спільна заготовка
// для бойових тестів: усі вони про те, ЯК зараховується шкода, а не про геометрію.
func wardenTestGame(t *testing.T) (*Game, *Brain) {
	t.Helper()
	savedRoster, savedMap := unitRoster, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	t.Cleanup(func() { unitRoster, tileMap = savedRoster, savedMap })

	w := ConfigWarden
	w.Count, w.WeightsFile = 1, "test_warden_never_exists.json"
	unitRoster = []UnitConfig{w}

	// Тіла ПЕРЕТИНАЮТЬСЯ, але центри РІЗНІ. Точка-в-точку не годиться: unitTo при
	// нульовій відстані не дає напрямку, і тест «біг не ранить» проходив би з
	// неправильної причини — не тому, що правило працює, а тому, що шкода взагалі
	// не рахувалась. Саме це й сталось із першою версією цього тесту.
	g := &Game{difficulty: 1.0, units: newUnits()}
	g.player = Pixel{
		X: g.units[0].X - pixelSize*0.4, Y: g.units[0].Y,
		HP: 100, MaxHP: 100, Faction: factionPlayer,
	}
	b := g.units[0].Brain
	if b == nil {
		t.Fatal("стражник без мозку")
	}
	return g, b
}

// TestDashDamagesOnlyInActivePhase — головний регресійний запобіжник нової механіки.
//
// Було: шкода — ПОБІЧНИЙ ЕФЕКТ швидкості (поріг 0.6×5.0 = 3.0, досяжний просто на
// бігу). Стало: шкода — РЕЗУЛЬТАТ рішення вдарити. Саме тут зникає домінантна
// стратегія «швидко кататись і таранити», і саме тут зʼявляється те, що агент може
// прочитати.
//
// Тест пінить обидві половини, бо кожна окремо ламається молча:
//   - біг на ПОВНІЙ швидкості не ранить (інакше замах — декорація);
//   - ривок ранить І ВЧИТЬ (dmgTaken зростає — на відміну від ульти, яку ми з
//     навчання виключали саме через невідворотність; ривок відворотний, тож мусить
//     повернутись у нагороду).
func TestDashDamagesOnlyInActivePhase(t *testing.T) {
	g, b := wardenTestGame(t)

	// 1. Біг на повній швидкості просто в юніта — раніше це був удар.
	hpBefore := g.units[0].HP
	g.player.VelX, g.player.VelY = playerBaseSpeed, 0
	g.player.DashPhase = dashIdle
	g.resolveImpacts()
	if g.units[0].HP != hpBefore {
		t.Errorf("біг завдав шкоди (%d → %d): таран мусить бути НАВМИСНИМ, "+
			"інакше замах ні на що не впливає", hpBefore, g.units[0].HP)
	}
	if b.dmgTaken != 0 {
		t.Errorf("біг потрапив у нагороду: dmgTaken %d", b.dmgTaken)
	}

	// 2. Активна фаза ривка — навіть без швидкості: фаза І Є атака.
	g.units[0].InvulnTimer = 0
	hpBefore = g.units[0].HP
	g.player.VelX, g.player.VelY = 0, 0
	g.player.DashPhase = dashPhaseActive
	g.resolveImpacts()
	if g.units[0].HP != hpBefore-dashDamage {
		t.Errorf("ривок не завдав %d шкоди: %d → %d", dashDamage, hpBefore, g.units[0].HP)
	}
	if b.dmgTaken != dashDamage {
		t.Errorf("ривок НЕ потрапив у нагороду: dmgTaken %d, чекали %d.\n"+
			"  Ульту ми виключали, бо від неї не можна ухилитись. Від ривка МОЖНА, "+
			"тож він мусить учити.", b.dmgTaken, dashDamage)
	}
}

// TestAimStylesAreMutuallyExclusive — несуча властивість усього заміру стилів.
//
// Задача про передбачення не порожня ЛИШЕ якщо контрзаходи взаємовиключні: те, що рятує
// від одного стилю, мусить ловити від іншого. Інакше сліпий агент знайде одну політику
// проти обох, і знання стилю не дасть нічого — саме на цьому померла попередня
// конструкція «ранній/пізній замах» (виграш оракула +0.0 за матрицею).
//
// Перевіряємо буквально: цілі, що рухається, два стилі мусять цілити по РІЗНІ БОКИ від
// неї, і кожен мусить бити свій варіант її подальшого руху.
func TestAimStylesAreMutuallyExclusive(t *testing.T) {
	saved := dashStyleMirror
	defer func() { dashStyleMirror = saved }()

	from := &Pixel{X: 100, Y: 300}
	// Ціль праворуч від нападника, ЛЕТИТЬ угору (−Y).
	target := &Pixel{X: 300, Y: 300, VelY: -0.6}

	dashStyleMirror = false
	lx, ly := dashAimAt(from, target)
	dashStyleMirror = true
	mx, my := dashAimAt(from, target)

	if ly >= 0 {
		t.Errorf("ВЕДУЧИЙ цілить не вперед по руху цілі: dy = %.1f, чекали відʼємне", ly)
	}
	if my <= 0 {
		t.Errorf("ДЗЕРКАЛЬНИЙ цілить не в бік розвороту: dy = %.1f, чекали додатне", my)
	}
	if lx <= 0 || mx <= 0 {
		t.Errorf("обидва стилі мусять дивитись У БІК цілі по X: %.1f, %.1f", lx, mx)
	}

	// Відстань між точками прицілу мусить перевищувати КОРПУС — інакше промах
	// несправжній: ривок зачепить ціль обома стилями, і вибір нічого не вирішує.
	sep := 2 * dashLeadFrames * 0.6 // ±lead від позиції цілі
	if sep <= pixelSize {
		t.Errorf("точки прицілу розходяться лише на %.0fpx при корпусі %d — промах "+
			"несправжній, обидва стилі влучатимуть", sep, pixelSize)
	}
	t.Logf("розведення точок прицілу %.0fpx при корпусі %d", sep, pixelSize)
}

// TestOracleTakesSlot14AndHidesAim — Фаза 0 міняє перцепцію рівно у двох місцях.
func TestOracleTakesSlot14AndHidesAim(t *testing.T) {
	savedMap := tileMap
	saved := [3]bool{featDashStyle, featDashAim, dashStyleMirror}
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() {
		featDashStyle, featDashAim, dashStyleMirror = saved[0], saved[1], saved[2]
		tileMap = savedMap
	}()

	player := &Pixel{X: 300, Y: 300}
	startDash(player, 1, 0)
	for i := 0; i < dashWindup/2; i++ {
		advanceDash(player)
	}
	unit := func() *Pixel { return &Pixel{X: 340, Y: 300, HP: 5, MaxHP: 5, Cfg: ConfigWarden} }

	// Телеграф БЕЗ прицілу: значення мусить бути суто прогресом, тобто ДОДАТНИМ навіть
	// коли ривок летить геть. Інакше агент читає напрямок і передбачати нема чого.
	featDashAim = false
	away := &Pixel{X: 300, Y: 300}
	startDash(away, -1, 0)
	for i := 0; i < dashWindup/2; i++ {
		advanceDash(away)
	}
	if v := GatherInputs(unit(), away)[inDashAtMe]; v <= 0 {
		t.Errorf("телеграф без прицілу дав %.3f для ривка ГЕТЬ — напрямок усе ще протікає", v)
	}

	// Оракул займає слот 14 і несе САМЕ стиль.
	featDashStyle = true
	dashStyleMirror = false
	if v := GatherInputs(unit(), player)[inDashOpen]; v != 0 {
		t.Errorf("ведучий стиль дав %.3f, чекали 0", v)
	}
	dashStyleMirror = true
	if v := GatherInputs(unit(), player)[inDashOpen]; v != 1 {
		t.Errorf("дзеркальний стиль дав %.3f, чекали 1", v)
	}
	// Без оракула слот 14 повертається до вікна покарання.
	featDashStyle = false
	for player.DashPhase != dashPhaseRecovery {
		advanceDash(player)
	}
	if v := GatherInputs(unit(), player)[inDashOpen]; v <= 0 {
		t.Errorf("без оракула слот 14 мусить нести вікно покарання, дав %.3f", v)
	}
}

// TestRewardFitsUnderErrorClip — найбільша штатна подія мусить лізти під кліп помилки.
//
// tdUpdate обрізає TD-помилку до [−1,1]. Поки нагорода за влучний ривок була
// 8 × (−2) = −16, а за таран 1 × (−2) = −2, ОБІ події давали градієнт −1: агент не міг
// відрізнити те, від чого варто ухилятись, від того, чим варто розмінюватись. Уся
// виведена умова «ухилятись у 8 разів вигідніше» була невидима для навчання, і перший
// A/B по телеграфу дав рівно нуль саме через це.
//
// Тест тримає інваріант: підняв dashDamage або нагороду — перевір, чи ще влазиш.
func TestRewardFitsUnderErrorClip(t *testing.T) {
	worst := float32(dashDamage) * -rewardDamageTaken
	if worst > 1 {
		t.Errorf("влучний ривок дає нагороду %.2f, а кліп помилки ±1 — усе вище %g "+
			"сплющується в одне значення, і РІЗНИЦЯ МІЖ ПОДІЯМИ зникає.\n"+
			"  Треба rewardDamageTaken ≥ −%.3f (зараз %.3f) або менший dashDamage.",
			worst, 1.0, 1.0/float32(dashDamage), rewardDamageTaken)
	}
	// І навпаки: якщо найбільша подія на порядок МЕНША за кліп, ми задарма втратили
	// швидкість навчання — кліп тоді ні на що не впливає, а градієнти дрібні.
	if worst < 0.2 {
		t.Errorf("влучний ривок дає лише %.3f при доступному кліпі 1.0 — градієнти "+
			"дрібніші, ніж могли б бути", worst)
	}
}

// TestDodgeIsPhysicallyPossible — умова існування розвʼязку, а не тюнінг.
//
// Це той самий прорахунок на три рядки, якого нам забракло раніше: ми зняли ЧОТИРИ
// заміри по стражнику (γ, горизонт, памʼять, темп бою) і отримали чотири нулі, бо при
// гравцевих 5.0 проти його 0.6 у нього було 5 кадрів і 12% корпусу — політики, яка
// ухиляється, просто НЕ ІСНУВАЛО. Гіперпараметри тут ні до чого.
//
// Тепер це перевіряється машиною, а не пам'яттю про урок:
//
//	замах  ≥ корпус / швидкість  →  можна зійти з лінії
//	відхід ≥ корпус / швидкість  →  можна повернутись і покарати
//
// Без другої нерівності «ідеальна» політика — вічно тікати з нульовою нагородою:
// пацифіст, а не боєць.
func TestDodgeIsPhysicallyPossible(t *testing.T) {
	// Три доданки, а не один. Голої геометрії (25/0.6 = 41.7) НЕ досить: ε їсть частину
	// кадрів, а ознака замаху наростає з нуля, тож агент не може почати з кадру 0.
	// Із самою геометрією 45 кадрів «проходили» тест і не працювали б у ділі.
	const reactionLag = 0.25 // частка замаху, яку агент витрачає на розпізнавання
	eff := ConfigWarden.MaxSpeed * (1 - qEpsilonConst*7.0/8.0)
	need := float32(pixelSize) / eff / (1 - reactionLag)
	if float32(dashWindup) < need {
		t.Errorf("dashWindup = %d < %.1f кадрів: стражник (швидкість %.2f) не встигає зійти "+
			"з лінії удару на корпус (%d px) — УХИЛЕННЯ фізично неможливе, і жодне "+
			"навчання цього не виправить", dashWindup, need, ConfigWarden.MaxSpeed, pixelSize)
	}
	if float32(dashRecovery) < need {
		t.Errorf("dashRecovery = %d < %.1f кадрів: той, хто ухилився, не встигає повернутись — "+
			"оптимальна політика стає «тікати вічно» з нульовою нагородою", dashRecovery, need)
	}

	// ПОКАРАННЯ обмежує dashActive ЗВЕРХУ, і цю межу я спершу пропустив: при
	// dashActive = 10 ривок відносив гравця на 100px, а стражник за відхід криє 36px.
	// Вікно покарання існувало лише на папері, і замір це показав («завдано» 1.7 → 0.8).
	reach := float32(dashActive) * playerBaseSpeed * dashSpeedMulti
	canClose := float32(dashRecovery) * eff
	if reach-pixelSize > canClose {
		t.Errorf("ривок відносить гравця на %.0fpx, стражникові треба закрити %.0fpx, "+
			"а за %d кадрів відходу він криє лише %.0fpx — ПОКАРАННЯ фізично неможливе.\n"+
			"  Треба dashActive ≤ %.0f (зараз %d) АБО довший відхід.",
			reach, reach-pixelSize, dashRecovery, canClose,
			(canClose+pixelSize)/(playerBaseSpeed*dashSpeedMulti), dashActive)
	}
}

// TestDodgingBeatsCamping — ухилення мусить бути ВИГІДНІШИМ за розмін.
//
// Юніт ранить гравця дотиком на швидкості, тобто заробляє НЕПЕРЕРВНО, поки
// тримається біля нього (стеля — один удар на impactInvuln). За час між двома
// ривками він заробляє тим більше, чим ця пауза довша:
//
//	заробіток за цикл атаки:  impactDamage × dashCadence / impactInvuln
//	втрата від пропущеного:   dashDamage
//
// Якщо перше більше, оптимальна політика — стояти й розмінюватись, і ми б виміряли
// розмінника, а не ухильника. Саме це вийшло б при dashDamage = 5, і саме на цьому
// я спіймався: узяв за каденцію мінімальний цикл (100) замість зміряних 230.
//
// ⚠️ dashCadence — ЗМІРЯНЕ число. Змінивши dashActive, швидкість гравця або
// benchDashRange, його треба переміряти, інакше цей тест втратить силу.
func TestDodgingBeatsCamping(t *testing.T) {
	earn := float32(impactDamage) * float32(dashCadence) / float32(impactInvuln)
	lose := float32(dashDamage)
	if lose <= earn {
		t.Errorf("розмін вигідніший за ухилення: за цикл атаки (%d кадрів) юніт заробляє "+
			"%.1f, а пропущений ривок коштує лише %.1f.\n"+
			"  Треба dashDamage > %.1f (зараз %d), АБО коротша каденція, "+
			"АБО менший impactDamage.", dashCadence, earn, lose, earn, dashDamage)
	}
	if dashCycle > dashCadence {
		t.Errorf("dashCadence = %d менша за мінімальний цикл %d — число нереальне",
			dashCadence, dashCycle)
	}
}

// TestDashDirectionLocksAtWindupStart — напрямок замикається на ПОЧАТКУ замаху.
//
// Це різниця між реакцією і вгадуванням. Якби напрямок обирався в момент випуску,
// гравець дивився б, куди відійшов юніт, і націлювався туди — вийшла б гра
// «камінь-ножиці-папір», у якій оптимальна політика ВИПАДКОВА. Випадковість не
// вивчається, і замір знову був би про ніщо.
func TestDashDirectionLocksAtWindupStart(t *testing.T) {
	p := &Pixel{}
	if !startDash(p, 1, 0) {
		t.Fatal("ривок не почався")
	}
	dx, dy := p.DashDirX, p.DashDirY

	// Спроба перецілитись у будь-якій фазі мусить бути відхилена.
	for i := 0; i < dashCycle; i++ {
		if startDash(p, 0, 1) {
			t.Fatalf("кадр %d (фаза %d): напрямок перезадали — телеграф став брехнею",
				i, p.DashPhase)
		}
		if p.DashDirX != dx || p.DashDirY != dy {
			t.Fatalf("кадр %d: напрямок поїхав %.2f,%.2f → %.2f,%.2f",
				i, dx, dy, p.DashDirX, p.DashDirY)
		}
		advanceDash(p)
	}
	if p.DashPhase != dashIdle {
		t.Errorf("після %d кадрів фаза %d, чекали dashIdle", dashCycle, p.DashPhase)
	}
}

// TestDashPhasesRunInOrder — машина фаз проходить замах → ривок → відхід і саме
// стільки кадрів, скільки обіцяють константи. Шкода є ЛИШЕ в середній фазі, тож
// зсув на кадр тут — це зсув вікна, у яке агент мусить укластися.
func TestDashPhasesRunInOrder(t *testing.T) {
	p := &Pixel{}
	startDash(p, 1, 0)

	want := make([]int, 0, dashCycle)
	for i := 0; i < dashWindup; i++ {
		want = append(want, dashPhaseWindup)
	}
	for i := 0; i < dashActive; i++ {
		want = append(want, dashPhaseActive)
	}
	for i := 0; i < dashRecovery; i++ {
		want = append(want, dashPhaseRecovery)
	}
	for i, w := range want {
		if p.DashPhase != w {
			t.Fatalf("кадр %d: фаза %d, чекали %d", i, p.DashPhase, w)
		}
		advanceDash(p)
	}

	// Прогрес замаху — вхідна ознака мозку: мусить рости 0→1, інакше агент бачить шум.
	p2 := &Pixel{}
	startDash(p2, 1, 0)
	if got := dashWindupProgress(p2); got != 0 {
		t.Errorf("на початку замаху прогрес %.3f, чекали 0", got)
	}
	for i := 0; i < dashWindup-1; i++ {
		advanceDash(p2)
	}
	if got := dashWindupProgress(p2); got < 0.9 {
		t.Errorf("у кінці замаху прогрес %.3f, чекали ≈1", got)
	}
	advanceDash(p2) // перейшли в ривок
	if got := dashWindupProgress(p2); got != 0 {
		t.Errorf("у фазі ривка прогрес %.3f, чекали 0 (ознака описує лише ЗАМАХ)", got)
	}
}

// TestWindupStopsThePlayer — замах мусить бути ВИДИМИМ, і зупинка це його підпис.
//
// Перевіряємо через updatePlayer, а не через клавіатуру: handlePlayerInput читає
// ebiten, якого в тесті немає. Важливо саме те, що фізика не дає рухатись у замаху
// незалежно від того, звідки взялась швидкість, — інакше скриптований гравець на
// стенді «читерив» би, рухаючись у замаху, і телеграф став би брехнею.
func TestWindupStopsThePlayer(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	g := &Game{difficulty: 1.0}
	g.player = Pixel{X: 400, Y: 400, HP: 10, MaxHP: 10, VelX: playerBaseSpeed}
	g.player.resetFur()
	startDash(&g.player, 1, 0)

	g.updatePlayer()
	if g.player.VelX != 0 || g.player.VelY != 0 {
		t.Errorf("у замаху гравець рухається (%.2f, %.2f) — телеграф не видно",
			g.player.VelX, g.player.VelY)
	}

	// Доганяємо до активної фази: там швидкість мусить бути ПІДНЯТА над звичайною.
	for g.player.DashPhase == dashPhaseWindup {
		g.updatePlayer()
	}
	if g.player.DashPhase != dashPhaseActive {
		t.Fatalf("фаза %d після замаху", g.player.DashPhase)
	}
	speed := g.player.VelX
	if speed <= playerBaseSpeed {
		t.Errorf("швидкість ривка %.2f не перевищує звичайну %.2f — ривок не дістане "+
			"на dashActive×швидкість, і замах не має сенсу", speed, float32(playerBaseSpeed))
	}

	// Удар ПІД ЧАС ривка не мусить множити стелю вдруге. Дві незалежні механіки
	// підняття (віддача й ривок) перемножувались і давали 50px/кадр — телепорт через
	// пів екрана. Беремо більшу з двох, а не добуток.
	g.player.KnockTimer = knockFrames
	g.updatePlayer()
	if got := g.player.VelX; got > playerBaseSpeed*dashSpeedMulti+0.01 {
		t.Errorf("швидкість ривка після удару %.2f перевищує %.2f: стелі перемножились",
			got, playerBaseSpeed*dashSpeedMulti)
	}
}

// TestCombatInputsOnlyForCombatTypes — ознаки бою НЕ мусять зʼявлятись у рою.
//
// Це не мікрооптимізація, а захист ЗАМІРІВ. Рій має щільну нагороду за наближення,
// бій його не стосується — для нього невразливість і замах були б шумом. А головне:
// поки його вхід біт-у-біт той самий, усі записані базові лінії (памʼять 72+72, свіпи
// зору й горизонту, важіль BPTT) далі порівнюються з новими числами.
//
// Ознака, потрібна ОДНОМУ типу, не мусить знецінювати заміри всіх інших — саме тому
// ми й не пішли шляхом «додати всім, ваги все одно перевчимо».
func TestCombatInputsOnlyForCombatTypes(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	// Гравець у розпалі замаху, націленого просто в юніта, і юніт у невразливості —
	// тобто ОБІ ознаки мали б бути ненульові, якби тип їх отримував.
	player := &Pixel{X: 300, Y: 300, HP: 10, MaxHP: 10}
	startDash(player, 1, 0)
	for i := 0; i < dashWindup/2; i++ {
		advanceDash(player)
	}

	mk := func(cfg UnitConfig) [baseInputs]float32 {
		u := &Pixel{X: 340, Y: 300, HP: 5, MaxHP: 5, Cfg: cfg}
		return GatherInputs(u, player)
	}

	swarm := mk(ConfigLearner)
	if swarm[inDashOpen] != 0 || swarm[inDashAtMe] != 0 {
		t.Errorf("рій отримав бойові ознаки: [14]=%.3f [15]=%.3f — базові лінії "+
			"переслідування знецінені", swarm[inDashOpen], swarm[inDashAtMe])
	}

	warden := mk(ConfigWarden)
	if warden[inDashAtMe] <= 0 {
		t.Errorf("стражник не бачить замаху в себе: [15]=%.3f", warden[inDashAtMe])
	}

	// Друга половина: вікно покарання. Доганяємо гравця до фази відходу.
	for player.DashPhase != dashPhaseRecovery {
		advanceDash(player)
	}
	rec := mk(ConfigWarden)
	if rec[inDashOpen] <= 0 {
		t.Errorf("стражник не бачить вікна покарання: [14]=%.3f", rec[inDashOpen])
	}
	if rec[inDashAtMe] != 0 {
		t.Errorf("у відході ознака замаху %.3f, чекали 0", rec[inDashAtMe])
	}
	if mk(ConfigLearner)[inDashOpen] != 0 {
		t.Error("рій побачив вікно покарання")
	}

	// Вимикачі для A/B: без них не поставити питання «чи ознака взагалі щось дає».
	// Перевіряємо, що вони справді глушать, і що кожен глушить ЛИШЕ своє.
	saved := [2]bool{featDashOpen, featDashAtMe}
	defer func() { featDashOpen, featDashAtMe = saved[0], saved[1] }()

	featDashOpen, featDashAtMe = false, false
	if off := mk(ConfigWarden); off[inDashOpen] != 0 || off[inDashAtMe] != 0 {
		t.Errorf("вимикачі не діють: [14]=%.3f [15]=%.3f", off[inDashOpen], off[inDashAtMe])
	}
	featDashOpen = true
	if one := mk(ConfigWarden); one[inDashOpen] == 0 || one[inDashAtMe] != 0 {
		t.Errorf("вимикачі не незалежні: [14]=%.3f [15]=%.3f", one[inDashOpen], one[inDashAtMe])
	}
}

// TestTelegraphCarriesTimingAndAim — одне число мусить нести ОБА сенси.
//
// Вільний слот лишився один (подати напрямок ривка окремо означало б baseInputs 16→17
// і знецінення всіх ваг), тож ознака = прогрес замаху × проєкція напрямку ривка на
// напрямок «від гравця до мене». Перевіряємо, що не втратився ні таймінг, ні приціл:
// якщо ознака реагує лише на одне з двох, агент або ухилятиметься від чужих ривків,
// або не знатиме, коли саме.
func TestTelegraphCarriesTimingAndAim(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	unit := func() *Pixel {
		return &Pixel{X: 340, Y: 300, HP: 5, MaxHP: 5, Cfg: ConfigWarden}
	}
	// Юніт праворуч від гравця: ривок (+1,0) летить У нього, (−1,0) — від нього.
	feat := func(dirX, dirY float32, frames int) float32 {
		p := &Pixel{X: 300, Y: 300, HP: 10, MaxHP: 10}
		if dirX != 0 || dirY != 0 {
			startDash(p, dirX, dirY)
			for i := 0; i < frames; i++ {
				advanceDash(p)
			}
		}
		return GatherInputs(unit(), p)[inDashAtMe]
	}

	if v := feat(0, 0, 0); v != 0 {
		t.Errorf("без замаху ознака %.3f, чекали 0", v)
	}
	early, late := feat(1, 0, 2), feat(1, 0, dashWindup-2)
	if !(late > early) {
		t.Errorf("ознака не несе ТАЙМІНГ: рано %.3f, пізно %.3f — агент не дізнається, "+
			"коли саме прилетить", early, late)
	}
	if away := feat(-1, 0, dashWindup-2); away >= 0 {
		t.Errorf("ознака не несе ПРИЦІЛ: ривок У ПРОТИЛЕЖНИЙ бік дав %.3f, чекали "+
			"відʼємне — інакше агент ухилятиметься від чужих атак", away)
	}
	if side := feat(0, 1, dashWindup-2); side > 0.2 || side < -0.2 {
		t.Errorf("ривок ПЕРПЕНДИКУЛЯРНО дав %.3f, чекали ≈0", side)
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

// TestRespawnRevivesSameBrain — респаун мусить ОЖИВЛЯТИ той самий юніт, а не створювати
// нового.
//
// Це не оптимізація. m.agents у метриках ключується вказівником на Brain, тож новий
// Brain на кожну смерть роздував би «ag» до кількості СМЕРТЕЙ замість кількості юнітів.
// Ми вже ловили це на рестартах: один стражник після трьох смертей читався як «4 ag».
//
// Плюс перевіряємо hasPrev=false: без цього наступна нагорода порівняла б стан ПІСЛЯ
// відродження зі станом ПЕРЕД смертю, і в буфер ліг би перехід через межу смерті.
func TestRespawnRevivesSameBrain(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { unitRoster, tileMap = savedRoster, savedMap }()

	cfg := ConfigLearner
	cfg.Count, cfg.WeightsFile, cfg.Respawns = 1, "test_respawn_never_exists.json", 2
	unitRoster = []UnitConfig{cfg}

	g := &Game{difficulty: 1.0, units: newUnits()}
	brainBefore := g.units[0].Brain
	netBefore := brainBefore.net
	postX, postY := g.units[0].SpawnX, g.units[0].SpawnY

	// Відносимо юніта від поста й «вбиваємо».
	g.units[0].X, g.units[0].Y = postX+300, postY+300
	g.units[0].HP = 0
	g.units[0].Brain.hasPrev = true

	g.handleDeadUnits()

	if len(g.units) != 1 {
		t.Fatalf("юніт зник, хоч мав %d повернень: лишилось %d", cfg.Respawns, len(g.units))
	}
	if g.units[0].Brain != brainBefore {
		t.Error("створено НОВИЙ Brain — метрики почнуть рахувати смерті замість юнітів")
	}
	if g.units[0].Brain.net != netBefore {
		t.Error("мережа підмінилась — навчання втрачено")
	}
	if g.units[0].HP != g.units[0].MaxHP {
		t.Errorf("HP не відновлено: %d з %d", g.units[0].HP, g.units[0].MaxHP)
	}
	if g.units[0].X != postX || g.units[0].Y != postY {
		t.Errorf("не повернувся на пост: (%.0f,%.0f) замість (%.0f,%.0f)",
			g.units[0].X, g.units[0].Y, postX, postY)
	}
	if g.units[0].Brain.hasPrev {
		t.Error("hasPrev не скинуто → наступна нагорода зшила б стани через межу смерті")
	}
	if g.units[0].RespawnsLeft != cfg.Respawns-1 {
		t.Errorf("лічильник повернень: %d, очікували %d", g.units[0].RespawnsLeft, cfg.Respawns-1)
	}

	// Повернення закінчуються → юніт зникає.
	for i := 0; i < 5; i++ {
		g.units[0].HP = 0
		g.handleDeadUnits()
		if len(g.units) == 0 {
			break
		}
	}
	if len(g.units) != 0 {
		t.Error("юніт не зник після витрачених повернень")
	}
}

// TestInfiniteRespawnKeepsPopulation — негативне значення = безкінечно.
// Потрібно не для гри, а для ЗАМІРІВ: стала популяція означає, що знаменник метрик не
// пливе, і прогін, де пощастило вижити, порівнюється з прогоном, де юнітів вибили.
func TestInfiniteRespawnKeepsPopulation(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { unitRoster, tileMap = savedRoster, savedMap }()

	cfg := ConfigLearner
	cfg.Count, cfg.WeightsFile, cfg.Respawns = 3, "test_inf_never_exists.json", -1
	unitRoster = []UnitConfig{cfg}

	g := &Game{difficulty: 1.0, units: newUnits()}
	for round := 0; round < 20; round++ {
		for i := range g.units {
			g.units[i].HP = 0
		}
		g.handleDeadUnits()
		if len(g.units) != 3 {
			t.Fatalf("раунд %d: населення %d замість 3 — безкінечний респаун не працює",
				round, len(g.units))
		}
	}
}

// TestDeathCostsReward — смерть мусить доходити до нагороди.
//
// Була безкоштовною: юніта видаляли одразу, наступного Step він не отримував, і
// dmgTaken від фатального удару ніколи не ставав −2. Тобто в нагороді не існувало
// причини не вмирати — а для стражника, у якого бойова нагорода ЄДИНЕ джерело сигналу,
// це означало, що половина уроку зникала.
func TestDeathCostsReward(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { unitRoster, tileMap = savedRoster, savedMap }()

	cfg := ConfigWarden
	cfg.Count, cfg.WeightsFile, cfg.Respawns = 1, "test_death_never_exists.json", 1
	unitRoster = []UnitConfig{cfg}

	g := &Game{difficulty: 1.0, units: newUnits()}
	b := g.units[0].Brain
	before := b.net.replayLen()

	// Стан «жив і щойно діяв», фатальний удар уже зарахований у dmgTaken.
	b.hasPrev = true
	b.prevAction = 2
	b.dmgTaken = impactDamage
	g.units[0].HP = 0

	g.handleDeadUnits()

	if b.net.replayLen() != before+1 {
		t.Fatalf("термінальний перехід не потрапив у буфер: %d → %d", before, b.net.replayLen())
	}
	// Останній записаний перехід — саме термінальний і з відʼємною нагородою.
	last := b.net.replay[(b.net.replayHead-1+qReplaySize)%qReplaySize]
	if !last.terminal {
		t.Error("перехід не позначено terminal → ціль Беллмана додасть γ·maxQ майбутнього, якого немає")
	}
	if last.r >= 0 {
		t.Errorf("смерть не коштувала нагороди: r %.3f", last.r)
	}
}

// TestTelegraphReachesAgentInPlay — ЗАПОБІЖНИК ПРОТИ ХИБНОГО НУЛЯ.
//
// Перший замір A/B дав рівно нуль: 12.4% / 12.4% / 12.4% / 11.9%, медіани збіглись до
// десятої. Такий плоский результат однаково узгоджений із «підказка не допомагає» і з
// «підказка не доходить», а це два протилежні висновки. Розрізнити їх можна лише
// перевіркою водопроводу, і зробити її треба ДО того, як писати висновок у доки.
//
// Тест крутить справжній безголовий цикл зі скриптованим гравцем і дивиться, що
// НАСПРАВДІ бачить агент у слотах 14-15.
func TestTelegraphReachesAgentInPlay(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	savedSeek, savedArena := benchDriveSeek, benchArena
	tileMap = [boidMapH][boidMapW]bool{}
	benchDriveSeek, benchArena = true, true
	defer func() {
		unitRoster, tileMap = savedRoster, savedMap
		benchDriveSeek, benchArena = savedSeek, savedArena
	}()

	w := ConfigWarden
	w.Count, w.WeightsFile, w.Respawns = 1, "", -1
	unitRoster = []UnitConfig{w}

	g := newBenchGame()
	drive := benchDriver(true, true)

	var nAtMe, nOpen, nWindup int
	var maxAtMe, maxOpen float32
	const frames = 4000
	for i := 0; i < frames; i++ {
		g.tickHeadless(drive, true)
		if g.player.DashPhase == dashPhaseWindup {
			nWindup++
		}
		in := GatherInputs(&g.units[0], &g.player)
		if in[inDashAtMe] != 0 {
			nAtMe++
			if v := in[inDashAtMe]; v > maxAtMe {
				maxAtMe = v
			}
		}
		if in[inDashOpen] != 0 {
			nOpen++
			if v := in[inDashOpen]; v > maxOpen {
				maxOpen = v
			}
		}
	}
	t.Logf("за %d кадрів: замах гравця %d кадрів; агент бачив [15] у %d кадрах (макс %.2f), "+
		"[14] у %d кадрах (макс %.2f)", frames, nWindup, nAtMe, maxAtMe, nOpen, maxOpen)

	if nWindup == 0 {
		t.Fatal("скриптований гравець НЕ замахується — замір був би про ніщо")
	}
	if nAtMe == 0 {
		t.Errorf("агент НІ РАЗУ не побачив замах, хоч гравець замахувався %d кадрів: "+
			"нуль у A/B означає зламаний водопровід, а не відсутність ефекту", nWindup)
	}
	if nOpen == 0 {
		t.Error("агент ні разу не побачив вікна покарання")
	}
}

// TestStyleOffsetIsRealAtWindup — [ДІАГНОСТИКА] чи стилі взагалі РІЗНІ на практиці.
//
// Зсув точки прицілу = швидкість цілі × dashLeadFrames. Якщо стражник у момент замаху
// майже стоїть, обидва стилі цілять в ОДНУ точку, і біт стилю не несе інформації —
// оракул нічого не дасть, скільки сідів не крути.
//
// Міряємо реальний розподіл швидкості стражника саме в кадри початку замаху.
func TestStyleOffsetIsRealAtWindup(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	savedSeek, savedArena := benchDriveSeek, benchArena
	tileMap = [boidMapH][boidMapW]bool{}
	benchDriveSeek, benchArena = true, true
	defer func() {
		unitRoster, tileMap = savedRoster, savedMap
		benchDriveSeek, benchArena = savedSeek, savedArena
	}()

	w := ConfigWarden
	w.Count, w.WeightsFile, w.Respawns = 1, "", -1
	unitRoster = []UnitConfig{w}
	g := newBenchGame()
	drive := benchDriver(true, true)

	var speeds []float64
	prev := dashIdle
	for i := 0; i < 30000; i++ {
		g.tickHeadless(drive, true)
		if g.player.DashPhase == dashPhaseWindup && prev != dashPhaseWindup && len(g.units) > 0 {
			u := &g.units[0]
			speeds = append(speeds, math.Hypot(float64(u.VelX), float64(u.VelY)))
		}
		prev = g.player.DashPhase
	}
	if len(speeds) == 0 {
		t.Fatal("жодного замаху за 30000 кадрів")
	}
	sort.Float64s(speeds)
	med := speeds[len(speeds)/2]
	// Щоб два стилі цілили в РІЗНІ точки, зсув мусить перевищити пів корпуса.
	need := float64(pixelSize) / 2 / dashLeadFrames
	var ok int
	for _, v := range speeds {
		if v >= need {
			ok++
		}
	}
	t.Logf("замахів %d | медіанна швидкість стражника %.3f | потрібно ≥%.3f | "+
		"розрізненних замахів %d%% | медіанний зсув прицілу %.1fpx (корпус %d)",
		len(speeds), med, need, 100*ok/len(speeds), med*dashLeadFrames, pixelSize)
	if 100*ok/len(speeds) < 50 {
		t.Errorf("лише %d%% замахів мають зсув понад пів корпуса — у решті стилі цілять "+
			"в ОДНУ точку, і біт стилю не несе інформації", 100*ok/len(speeds))
	}
}

// TestStyleCountersAreAntiDiagonal — [ГЕОМЕТРІЯ] чи справді контрзаходи взаємовиключні.
//
// Уся конструкція Фази 0 тримається на одному твердженні: те, що рятує від ВЕДУЧОГО,
// мусить ЛОВИТИ від ДЗЕРКАЛЬНОГО, і навпаки. Якщо таблиця не антидіагональна, задача
// порожня механічно, і жодне навчання її не врятує.
//
// Симулюємо чисту геометрію без мозку: гравець замахується й летить, стражник рухається
// із заданою сталою швидкістю. Перевіряємо, чи тіла перетнулись за активні кадри.
func TestStyleCountersAreAntiDiagonal(t *testing.T) {
	saved := dashStyleMirror
	defer func() { dashStyleMirror = saved }()

	// hit — чи влучить ривок, якщо стражник поводиться так, як задано keep.
	hit := func(mirror, keep bool) bool {
		dashStyleMirror = mirror
		player := &Pixel{X: 300, Y: 300}
		w := &Pixel{X: 330, Y: 300, VelX: 0, VelY: 0.6} // рухається ВНИЗ, поруч
		ax, ay := dashAimAt(player, w)
		startDash(player, ax, ay)

		vy := float32(0.6)
		if !keep {
			vy = -0.6 // розвернувся
		}
		for f := 0; f < dashWindup+dashActive; f++ {
			if player.DashPhase == dashPhaseActive {
				sp := float32(playerBaseSpeed) * dashSpeedMulti
				player.X += player.DashDirX * sp
				player.Y += player.DashDirY * sp
				if collides(player.X, player.Y, w.X, w.Y) {
					return true
				}
			}
			w.Y += vy
			advanceDash(player)
		}
		return false
	}

	type row struct {
		name string
		mirr bool
		keep bool
		want bool
	}
	cases := []row{
		{"ВЕДУЧИЙ × продовжує", false, true, true},       // цілив уперед — влучив
		{"ВЕДУЧИЙ × розвернувся", false, false, false},   // пішов проти прицілу — промах
		{"ДЗЕРКАЛЬНИЙ × продовжує", true, true, false},   // цілив назад — промах
		{"ДЗЕРКАЛЬНИЙ × розвернувся", true, false, true}, // пішов у приціл — влучив
	}
	bad := 0
	for _, c := range cases {
		got := hit(c.mirr, c.keep)
		mark := "OK"
		if got != c.want {
			mark, bad = "← НЕ ТАК", bad+1
		}
		t.Logf("%-28s влучив=%-5v чекали=%-5v %s", c.name, got, c.want, mark)
	}
	if bad > 0 {
		t.Errorf("таблиця НЕ антидіагональна (%d з 4 не так): контрзаходи не "+
			"взаємовиключні, задача порожня механічно", bad)
	}
}

// TestStyleCountersAcrossMotionAngles — [ДІАГНОСТИКА] чи антидіагональність тримається
// для ВСІХ напрямків руху цілі, чи лише для перпендикулярного.
//
// Перша перевірка (TestStyleCountersAreAntiDiagonal) брала стражника, що рухається
// ПЕРПЕНДИКУЛЯРНО до нападника, і дала ідеальну таблицю. Але справжній стражник
// АТАКУЄ, тобто рухається переважно НА гравця — радіально. Якщо для радіального руху
// обидва стилі влучають однаково, біт стилю не несе інформації саме в тих кадрах, які
// трапляються найчастіше, і оракул приречений на нуль.
func TestStyleCountersAcrossMotionAngles(t *testing.T) {
	saved := dashStyleMirror
	defer func() { dashStyleMirror = saved }()

	hit := func(mirror bool, vx, vy float32, keep bool) bool {
		dashStyleMirror = mirror
		player := &Pixel{X: 300, Y: 300}
		w := &Pixel{X: 330, Y: 300, VelX: vx, VelY: vy}
		ax, ay := dashAimAt(player, w)
		startDash(player, ax, ay)
		if !keep {
			vx, vy = -vx, -vy
		}
		for f := 0; f < dashWindup+dashActive; f++ {
			if player.DashPhase == dashPhaseActive {
				sp := float32(playerBaseSpeed) * dashSpeedMulti
				player.X += player.DashDirX * sp
				player.Y += player.DashDirY * sp
				if collides(player.X, player.Y, w.X, w.Y) {
					return true
				}
			}
			w.X += vx
			w.Y += vy
			advanceDash(player)
		}
		return false
	}

	// Стражник праворуч від гравця. Кут 0° = рух ПРЯМО НА гравця (радіально),
	// 90° = перпендикулярно, 180° = від гравця.
	t.Logf("%-22s %-12s %-12s %-12s %-12s %s", "напрямок руху",
		"вед×прод", "вед×розв", "дзерк×прод", "дзерк×розв", "інформативно?")
	informative := 0
	angles := []int{0, 30, 45, 60, 90, 120, 150, 180}
	for _, deg := range angles {
		rad := float64(deg) * math.Pi / 180
		// 0° = на гравця = −X (гравець ліворуч від стражника)
		vx := float32(-math.Cos(rad)) * 0.6
		vy := float32(math.Sin(rad)) * 0.6
		lk, lr := hit(false, vx, vy, true), hit(false, vx, vy, false)
		mk, mr := hit(true, vx, vy, true), hit(true, vx, vy, false)
		// Інформативно, якщо НАЙКРАЩА реакція залежить від стилю.
		bestLead := "продовж"
		if lk && !lr {
			bestLead = "розворот"
		}
		bestMirr := "продовж"
		if mk && !mr {
			bestMirr = "розворот"
		}
		ok := bestLead != bestMirr
		if ok {
			informative++
		}
		t.Logf("%3d° %-18s %-12v %-12v %-12v %-12v %v", deg, "", lk, lr, mk, mr, ok)
	}
	t.Logf("→ інформативних напрямків: %d з %d", informative, len(angles))
	if informative*2 < len(angles) {
		t.Errorf("біт стилю несе інформацію лише для %d з %d напрямків руху — "+
			"для решти обидва стилі дають ту саму найкращу реакцію",
			informative, len(angles))
	}
}

// TestWardenMotionAngleAtWindup — [ДІАГНОСТИКА] під яким кутом до гравця стражник
// РЕАЛЬНО рухається в момент замаху.
//
// TestStyleCountersAcrossMotionAngles показав, що біт стилю мовчить для радіального
// руху (0° і 180°) і говорить лише для 30-120°. Якщо реальний розподіл зосереджений
// біля 0°, оракул приречений на нуль — і саме це ми зміряли.
func TestWardenMotionAngleAtWindup(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	savedSeek, savedArena := benchDriveSeek, benchArena
	tileMap = [boidMapH][boidMapW]bool{}
	benchDriveSeek, benchArena = true, true
	defer func() {
		unitRoster, tileMap = savedRoster, savedMap
		benchDriveSeek, benchArena = savedSeek, savedArena
	}()
	w := ConfigWarden
	w.Count, w.WeightsFile, w.Respawns = 1, "", -1
	unitRoster = []UnitConfig{w}
	g := newBenchGame()
	drive := benchDriver(true, true)

	buckets := make([]int, 6) // 0-30,30-60,60-90,90-120,120-150,150-180
	prev, n := dashIdle, 0
	for i := 0; i < 40000; i++ {
		g.tickHeadless(drive, true)
		if g.player.DashPhase == dashPhaseWindup && prev != dashPhaseWindup && len(g.units) > 0 {
			u := &g.units[0]
			sp := math.Hypot(float64(u.VelX), float64(u.VelY))
			if sp > 0.05 {
				// кут між ВЛАСНОЮ швидкістю і напрямком НА гравця
				dx, dy := float64(g.player.X-u.X), float64(g.player.Y-u.Y)
				d := math.Hypot(dx, dy)
				if d > 0.001 {
					cos := (float64(u.VelX)*dx + float64(u.VelY)*dy) / (sp * d)
					deg := math.Acos(math.Max(-1, math.Min(1, cos))) * 180 / math.Pi
					b := int(deg / 30)
					if b > 5 {
						b = 5
					}
					buckets[b]++
					n++
				}
			}
		}
		prev = g.player.DashPhase
	}
	if n == 0 {
		t.Fatal("жодного замаху")
	}
	labels := []string{"0-30° НА гравця", "30-60°", "60-90°", "90-120°", "120-150°", "150-180° ВІД"}
	mute := 0
	for i, c := range buckets {
		flag := ""
		if i == 0 || i == 5 {
			flag = "  ← біт стилю МОВЧИТЬ"
			mute += c
		}
		t.Logf("  %-16s %4d  %3d%%%s", labels[i], c, 100*c/n, flag)
	}
	t.Logf("→ замахів, де стиль не несе інформації: %d%% (%d з %d)", 100*mute/n, mute, n)
}

// TestRewardChainCoverage — [ПЕРЕДУМОВА] яка частка кадрів належить ланцюжку нагороди.
//
// Ідея двох буферів (гарячий для ланцюжків, що привели до нагороди) має сенс ЛИШЕ якщо
// значна частина кадрів до жодного ланцюжка не належить — інакше «гарячим» стає майже
// все, і схема вироджується в один МЕНШИЙ буфер. Саме це сталось у драбині, де кожен
// епізод закінчувався нагородою.
//
// Тут міряємо покриття на СПРАВЖНЬОМУ стражнику зі скриптованим гравцем: для кожного
// кадру дивимось, чи є нагорода в наступних window кадрах.
func TestRewardChainCoverage(t *testing.T) {
	savedRoster, savedMap := unitRoster, tileMap
	savedSeek, savedArena := benchDriveSeek, benchArena
	tileMap = [boidMapH][boidMapW]bool{}
	benchDriveSeek, benchArena = true, true
	defer func() {
		unitRoster, tileMap = savedRoster, savedMap
		benchDriveSeek, benchArena = savedSeek, savedArena
	}()

	w := ConfigWarden
	w.Count, w.WeightsFile, w.Respawns = 1, "", -1
	unitRoster = []UnitConfig{w}
	g := newBenchGame()
	drive := benchDriver(true, true)

	const frames = 40000
	hasReward := make([]bool, frames)
	for i := 0; i < frames; i++ {
		before := 0
		if len(g.units) > 0 && g.units[0].Brain != nil {
			before = g.units[0].Brain.mDmgDealt + g.units[0].Brain.mDmgTaken
		}
		g.tickHeadless(drive, true)
		if len(g.units) > 0 && g.units[0].Brain != nil {
			if g.units[0].Brain.mDmgDealt+g.units[0].Brain.mDmgTaken > before {
				hasReward[i] = true
			}
		}
	}
	events := 0
	for _, v := range hasReward {
		if v {
			events++
		}
	}
	t.Logf("подій нагороди: %d на %d кадрів (%.1f на 1000)",
		events, frames, 1000*float64(events)/frames)
	t.Logf("%10s %14s %16s", "вікно", "покриття", "стиснення")
	for _, win := range []int{30, 65, 100, 200} {
		covered := 0
		for i := 0; i < frames; i++ {
			for j := i; j < i+win && j < frames; j++ {
				if hasReward[j] {
					covered++
					break
				}
			}
		}
		frac := 100 * float64(covered) / frames
		t.Logf("%10d %12.1f%% %14.1fx", win, frac, 100/frac)
	}
	t.Logf("ЧИТАННЯ: покриття близьке до 100%% → «гарячим» стає майже все,")
	t.Logf("         два буфери вироджуються в один менший, як у драбині")
}

// TestDodgeActionGrantsInvulnAndCosts — механіка девʼятої дії цілком.
//
// Перевіряємо ЧОТИРИ властивості, і кожна з них поодинці ламається молча:
//
//  1. ухилення дає невразливість;
//  2. поки воно триває, юніт НЕ прискорюється — це ціна, без якої «тиснути щойно
//     перезарядилось» стало б слабко домінантною сталою політикою;
//  3. кулдаун справді блокує повторне ухилення;
//  4. дія ухилення не читає вус (у неї немає напрямку) — інакше inWhisker0 + 8 = 13
//     читало б слот «видно гравця» як стіну.
func TestDodgeActionGrantsInvulnAndCosts(t *testing.T) {
	savedMap := tileMap
	tileMap = [boidMapH][boidMapW]bool{}
	defer func() { tileMap = savedMap }()

	u := &Pixel{X: 300, Y: 300, HP: 10, MaxHP: 10, Cfg: ConfigWarden}

	// 1) Невразливість.
	u.DodgeTimer = dodgeInvuln
	hp := u.HP
	attacker := &Pixel{X: 280, Y: 300, VelX: 5}
	applyImpactDamage(attacker, u, dashDamage)
	if u.HP != hp {
		t.Errorf("ухилення не захистило: %d → %d", hp, u.HP)
	}
	// А без нього — б'є.
	u.DodgeTimer = 0
	applyImpactDamage(attacker, u, dashDamage)
	if u.HP != hp-dashDamage {
		t.Errorf("без ухилення шкода не пройшла: %d → %d", hp, u.HP)
	}

	// 4) Вус для дії ухилення — нуль, а не слот «видно гравця».
	b := NewBrain()
	var st [brainInputs]float32
	st[inVisible] = 1      // якби індексація поїхала, це стало б «стіною»
	st[inWhisker0+2] = 0.7 // справжній вус напрямку 2
	if v := b.whiskerOf(st[:], actionDodge); v != 0 {
		t.Errorf("дія ухилення прочитала вус %.2f — індексація поїхала на слот %d",
			v, inWhisker0+actionDodge)
	}
	if v := b.whiskerOf(st[:], 2); v != 0.7 {
		t.Errorf("звичайна дія читає не той вус: %.2f", v)
	}
}

// TestDodgeIsAffordableAndDiscriminating — арифметика дії, а не її код.
//
// Ухилення має сенс лише якщо ТОЧНИЙ таймінг помітно кращий за спам. Інакше агент
// вивчить сталу політику «тиснути щойно перезарядилось» і нічого не читатиме — рівно
// та вада, через яку ухилення РУХОМ виявилось безкоштовним.
func TestDodgeIsAffordableAndDiscriminating(t *testing.T) {
	// Вікно мусить накрити активну фазу ривка із запасом на похибку таймінгу.
	if dodgeInvuln < dashActive*2 {
		t.Errorf("dodgeInvuln = %d замалий: активна фаза ривка %d кадрів, запасу на "+
			"похибку таймінгу майже немає", dodgeInvuln, dashActive)
	}
	// Спам покриває dodgeInvuln із кожних dodgeCooldown кадрів; точний таймінг — усе.
	spam := float64(dodgeInvuln) / float64(dodgeCooldown)
	if spam > 0.35 {
		t.Errorf("спам покриває %.0f%% часу — дискримінація замала, стала політика "+
			"«тиснути щойно перезарядилось» буде майже така сама добра, як читання", spam*100)
	}
	t.Logf("спам покриває %.0f%% часу проти 100%% у точного таймінгу → дискримінація %.1fx",
		spam*100, 1/spam)
	// І кулдаун не має бути довшим за каденцію атак: інакше на частину ривків
	// ухилення фізично недоступне, і задача знову стає нерозвʼязною.
	if dodgeCooldown > dashCadence {
		t.Errorf("dodgeCooldown = %d довший за каденцію атак %d — на частину ривків "+
			"ухилення просто недоступне", dodgeCooldown, dashCadence)
	}
}

// TestBodyVerticesHaveNoHole — регресія на візуальну ваду від девʼятої дії.
//
// Масив вершин був оголошений як [brainActions] (9), а заповнювався на brainWhiskers
// (8). Девʼята лишалась у ПОЧАТКУ КООРДИНАТ, і крива йшла через неї — контур тіла
// тягнувся в лівий верхній кут екрана через пів карти.
//
// Причина глибша за один масив: dirs8 теж був оголошений на brainActions і мовчки
// віддавав dirs8[8] = {0,0} замість того, щоб упасти. Тепер він рівно на 8, тож будь-яке
// індексування дією ухилення падає ОДРАЗУ й у правильному місці.
//
// Тест перевіряє геометрію, а не малювання: усі вершини мусять лежати БІЛЯ юніта.
func TestBodyVerticesHaveNoHole(t *testing.T) {
	if len(dirs8) != brainWhiskers {
		t.Fatalf("dirs8 має %d елементів, а напрямків %d — саме така розбіжність і "+
			"давала мовчазний {0,0}", len(dirs8), brainWhiskers)
	}

	p := Pixel{X: 900, Y: 500, BodyScale: 1}
	cx, cy := p.X+pixelSize/2, p.Y+pixelSize/2
	// З мозком і НЕРІВНОЮ Q — щоб деформація була максимальною.
	b := NewBrain()
	for i := range b.lastQ {
		b.lastQ[i] = float32(i) * 0.5
	}
	p.Brain = b

	vx, vy := bodyVertices(p, cx, cy, 1)
	maxR := float32(pixelSize) * (1 + bodyQStretch) * 1.5 // з великим запасом
	for i := range vx {
		dx, dy := vx[i]-cx, vy[i]-cy
		r := float32(math.Hypot(float64(dx), float64(dy)))
		if r > maxR {
			t.Errorf("вершина %d за %.0fpx від центра (стеля %.0f) — діра в контурі",
				i, r, maxR)
		}
		if vx[i] == 0 && vy[i] == 0 {
			t.Errorf("вершина %d у початку координат — незаповнений елемент масиву", i)
		}
	}
}

// TestDodgeBurstClearsTheLine — [ВІДКИД] стрибок мусить ЗНІМАТИ тіло з лінії удару.
//
// Суто часове ухилення мало вікно 20 кадрів із циклу 200 — 10% часу, стільки ж дає
// випадкове натискання. Зміряно: навчений блокував лише 16% ривків, ледве вибиваючись
// за випадок. Відкид розширює вікно, бо промах у таймінгу компенсується тим, що тіла
// вже немає на лінії.
func TestDodgeBurstClearsTheLine(t *testing.T) {
	// Загроза ліворуч, юніт праворуч від неї.
	threat := &Pixel{X: 300, Y: 300}
	e := &Pixel{X: 360, Y: 300, Cfg: ConfigWarden, VelY: 0.3} // хилиться вниз
	dodgeBurst(e, threat)

	// 1) Перпендикулярно: загроза по осі X, отже відкид мусить бути по Y.
	if math.Abs(float64(e.VelX)) > 0.01 {
		t.Errorf("відкид не перпендикулярний: VelX = %.2f, чекали ~0", e.VelX)
	}
	// 2) У той бік, куди юніт УЖЕ хилився.
	if e.VelY <= 0 {
		t.Errorf("відкид проти власного руху: VelY = %.2f при нахилі +0.3", e.VelY)
	}
	// 3) Швидкість — саме відкидна.
	sp := math.Hypot(float64(e.VelX), float64(e.VelY))
	if math.Abs(sp-dodgeDashSpeed) > 0.01 {
		t.Errorf("швидкість відкиду %.2f, чекали %.2f", sp, dodgeDashSpeed)
	}

	// 4) За час невразливості тіло мусить зійти щонайменше на КОРПУС.
	v, dist := float32(dodgeDashSpeed), float32(0)
	cover := -1
	for f := 0; f < dodgeInvuln; f++ {
		dist += v
		v *= damping
		if cover < 0 && dist >= pixelSize {
			cover = f + 1
		}
	}
	if dist < pixelSize {
		t.Errorf("за %d кадрів відкид дав %.1fpx при корпусі %d — лінія не звільняється",
			dodgeInvuln, dist, pixelSize)
	}
	t.Logf("корпус (%dpx) покрито за %d кадрів, усього %.1fpx за %d кадрів",
		pixelSize, cover, dist, dodgeInvuln)

	// 5) Вікно натискання мусить ПОМІТНО розширитись проти суто часового.
	oldWin, newWin := dodgeInvuln, dashWindup-cover
	if newWin <= oldWin {
		t.Errorf("вікно не розширилось: було %d кадрів, стало %d", oldWin, newWin)
	}
	t.Logf("вікно натискання: %d → %d кадрів (%.0f%% → %.0f%% циклу %d)",
		oldWin, newWin, 100*float64(oldWin)/dashCadence,
		100*float64(newWin)/dashCadence, dashCadence)
}

// TestBallsLagBehindTheBody — [КУЛЬКИ] уся анімація тримається на ВІДСТАВАННІ.
//
// Кулька підтягується до своєї точки під тілом лише на частку шляху за кадр, тож при
// русі юніта вона принципово не встигає. Якби вона трималась жорстко, не було б ані
// розгойдування на поворотах, ані відкиду назад при розгоні — тобто анімації взагалі.
//
// Заодно ловимо розʼїзд ballHome між спавном і оновленням: якби вони давали різні точки,
// кулька стрибала б на першому ж кадрі.
func TestBallsLagBehindTheBody(t *testing.T) {
	u := &Pixel{X: 500, Y: 500}
	u.resetFur()

	// На спавні кульки СТОЯТЬ у своїх домівках — без стрибка.
	cx, cy := u.X+pixelSize/2, u.Y+pixelSize/2
	for i := 0; i < ballCount; i++ {
		hx, hy := ballHome(i, cx, cy)
		if math.Abs(float64(u.Balls[i][0]-hx)) > 1e-4 || math.Abs(float64(u.Balls[i][1]-hy)) > 1e-4 {
			t.Errorf("кулька %d на спавні не в домівці: (%.2f, %.2f) проти (%.2f, %.2f)",
				i, u.Balls[i][0], u.Balls[i][1], hx, hy)
		}
	}

	// Різко зсуваємо тіло — кульки мусять ВІДСТАТИ, а не телепортуватись.
	before := u.Balls
	u.X += 40
	updateBalls(u)
	cx = u.X + pixelSize/2
	for i := 0; i < ballCount; i++ {
		hx, _ := ballHome(i, cx, cy)
		moved := u.Balls[i][0] - before[i][0]
		if moved <= 0 {
			t.Errorf("кулька %d не рушила за тілом зовсім: %.2f", i, moved)
		}
		if u.Balls[i][0] >= hx-1 {
			t.Errorf("кулька %d наздогнала домівку за ОДИН кадр (%.1f при цілі %.1f) — "+
				"відставання немає, анімації не буде", i, u.Balls[i][0], hx)
		}
	}

	// І все ж доганяє, якщо тіло стоїть: інакше кульки відірвались би назавжди.
	for f := 0; f < 200; f++ {
		updateBalls(u)
	}
	for i := 0; i < ballCount; i++ {
		hx, hy := ballHome(i, cx, cy)
		d := math.Hypot(float64(u.Balls[i][0]-hx), float64(u.Balls[i][1]-hy))
		if d > 0.5 {
			t.Errorf("кулька %d не наздогнала за 200 кадрів: відстань %.2f", i, d)
		}
	}
}
