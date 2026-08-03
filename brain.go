package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
)

// ==========================================================================
// НЕЙРОН-УЧЕНЬ v3: Q-LEARNING (REINFORCEMENT LEARNING) + WHISKERS.
//
// Що змінилось проти v2 (supervised MLP):
//   v2: вчитель казав ТОЧНУ відповідь (interceptDir) → мережа копіювала формулу.
//   v3: ВЧИТЕЛЯ НЕМАЄ. Агент діє і отримує лише reward (нагороду/штраф).
//       Стратегію він мусить відкрити САМ.
//
//   supervised:  "правильний рух = (0.83, 0.55)"      ← ми знаємо ціль
//   RL:          "ти зробив дію 3 → отримав +0.1"     ← знаємо лише оцінку
//
// Чому RL потрібен для стін: для обходу перешкод немає простої формули-вчителя.
// Зате reward тривіальний: врізався → -, наблизився → +. Агент сам зрозуміє,
// що в стіни врізатись погано, і навчиться їх обходити.
//
// [SHARED BRAIN] Поділ на Net + Brain:
//   Net   — сама мережа (ваги + target + буфер досвіду). Може бути СПІЛЬНОЮ.
//   Brain — «голова» одного ворога: указник на Net + ОСОБИСТА пам'ять агента.
//   Режим sharedBrain (main.go): усі учні ділять один Net → «вулик-розум».
//
// --------------------------------------------------------------------------
// ДВА ШЛЯХИ ПАМʼЯТІ (обираються прапорцем useGRU у рантаймі, див. Step/train):
//
//	brain_stack.go — FRAME-STACKING: вхід = стек stackFrames кадрів (64 числа),
//	                 вікно історії задане НАМИ. Історично перший підхід; лишається
//	                 як baseline для порівняння і як fallback.
//	brain_gru.go   — GRU: вхід = ОДИН кадр (16), «минуле» живе в прихованому стані
//	                 h, і мережа САМА вчиться, що тримати й як довго. НЕ дефолт:
//	                 програв стеку на замірах і просаджує TPS через BPTT.
//
// Тут, у brain.go, — лише СПІЛЬНЕ ядро обох: константи, Net/Brain, активації,
// ε-greedy, вуса й зір (whiskers, POMDP), anti-stuck, збереження ваг.
//
// Обидва шляхи потрібні для POMDP: агент бачить гравця лише поблизу й по прямій
// (localSight), тож без памʼяті він амнезик — забуває ціль, щойно вона за рогом.
// ==========================================================================

const (
	baseInputs  = 16 // ОДИН кадр стану (спільний розмір для ВСІХ типів мозку)
	stackFrames = 4  // [ПАМ'ЯТЬ] МІСТКІСТЬ стеку — задає розмір входу мережі

	brainInputs = baseInputs * stackFrames // повний вхід мережі (стек кадрів)

	// [ГЛИБИНА] Два прихованих шари — «зігнути, потім зігнути ще раз» (оріґамі).
	// Глибина ефективніше вичленяє складні залежності, ніж один ширший шар.
	brainHidden1 = 32 // 1-й прихований шар (згин простору входів)
	brainHidden2 = 16 // 2-й прихований шар (згин поверх згину)

	brainActions  = 8 // 8 напрямків руху (= кількість виходів Q)
	brainWhiskers = 8 // промені-сенсори стін

	// [RNN/GRU] Розмір рекурентного прихованого стану h (памʼять, яку мережа несе
	// між кадрами). Вмикається прапорцем useGRU: тоді вхід — ОДИН кадр (baseInputs),
	// а «минуле» живе в h, а не в стеку кадрів. Ваги GRU співіснують у Net поряд зі
	// стек-вагами (обираємо шлях у рантаймі) → перемкнути назад = нуль ризику.
	gruHidden = 32

	// [RNN крок 2] Довжина відрізка траєкторії для навчання крізь час (BPTT).
	// Рекурентна мережа вчиться на ПОСЛІДОВНОСТЯХ, а не окремих кадрах: щоб
	// «розгорнути» памʼять, треба seqLen поспіль кадрів ОДНОГО агента.
	seqLen        = 8                  // кадрів у НАВЧАЛЬНОМУ вікні (на них рахуємо loss/BPTT)
	seqBurnIn     = 4                  // кадрів ПРОГРІВУ перед вікном (лише forward, щоб h став реальним)
	seqTotal      = seqBurnIn + seqLen // усього кадрів у відрізку (12)
	seqReplaySize = 512                // місткість буфера відрізків
	seqBatch      = 8                  // відрізків на кадр у навчанні
	seqMinReplay  = 32                 // не вчимось, поки буфер не набрав стільки відрізків

	qLearnRate = 0.005 // швидкість навчання (RL шумніший за supervised → помірно)

	// [3] ε-greedy: частка ВИПАДКОВИХ дій (exploration). За замовчуванням ПОСТІЙНА
	// (qEpsilonConst). Якщо ввімкнути прапорець epsilonDecayEnabled (main.go) — ε
	// лінійно спадає max→min за qEpsilonDecay кроків (свіжий мозок досліджує →
	// з часом мисливець), але не нижче floor (qEpsilonMin).
	qEpsilonConst = 0.01  // постійна ε (коли спад ВИМКНЕНО) — поточний режим
	qEpsilonMax   = 0.30  // старт автоспаду (свіжий мозок)
	qEpsilonMin   = 0.03  // floor автоспаду — мінімум назавжди
	qEpsilonDecay = 25000 // кроків спаду max→min (~3.5 хв @120fps)

	// [DQN: стабілізатори нейро-Q-learning]
	qTargetSync = 1000 // кожні N оновлень копіюємо ваги в target-мережу
	qReplaySize = 4096 // розмір буфера досвіду (ring buffer)
	qBatch      = 16   // скільки випадкових переходів вчимо щокадру
	qMinReplay  = 200  // не вчимось, поки буфер не набрав стільки переходів

	brainMaxWeight = 5.0 // стеля ваг
	brainForce     = 0.3 // масштаб дії у прискорення

	whiskerRange = 110.0 // далекість «вусів» у px (~5 тайлів)
	whiskerStep  = 4.0   // крок променя при пошуку стіни

	// [ВБИВЦЯ] Нормалізатор відстані по лабіринту (у клітинках BFS). Ділимо на
	// ФІКСОВАНУ константу, а НЕ на maxDist поля: maxDist міняється при кожній
	// перебудові, тож та сама фізична відстань давала б різні числа на вході.
	flowDistNorm = 60.0

	// Масштаб reward підібраний так, щоб «хороший» кадр давав сигнал ~0.5,
	// а удар об стіну — помітний штраф. Замалий reward = TD-сигнал тоне в шумі.
	rewardCloserScale = 0.5  // нагорода за наближення до гравця (на px/кадр)
	rewardWallHit     = -1.0 // штраф за удар об стіну (по факту зіткнення)
	rewardNearWall    = -0.3 // [1] штраф за рух У БІК близької стіни (плавний градієнт обходу)

	// [БІЙ] Бойова нагорода — лише для мозків із CombatReward (вбивці). Рій живе
	// на старій, тонко налаштованій нагороді й лишається baseline-ом.
	//
	// Це РІДКІСНІ події (удар трапляється раз на багато кадрів), тому вони не
	// замінюють щільне «наближайся», а лягають ПОВЕРХ нього: щільна частина веде
	// агента до цілі, бойова каже, що робити, коли він уже там. Без щільної
	// складової рідкісний сигнал не вивчився б (перевірено на попередніх етапах).
	//
	// Відступ програмувати НЕ треба: штраф за отриману шкоду робить зависання
	// впритул після кидка невигідним → «вкусив і відскочив» виникає саме.
	rewardDamageDealt = 2.0  // за кожну одиницю завданої шкоди
	rewardDamageTaken = -2.0 // за кожну отриману
	rewardKill        = 5.0  // за добивання цілі

	// [2] anti-stuck «фрустрація»: коли агент застряг (б'ється в стіну АБО
	// притиснутий до неї без прогресу) — на frustrationFrames кадрів форсуємо
	// НАПРАВЛЕНИЙ вихід у найвідкритіший бік (escapeAction), а не випадковий смик.
	stuckHitInc       = 4    // +N за кадр з ударом об стіну
	stuckNoProgInc    = 1    // +N за кадр «біля стіни без наближення до гравця»
	stuckDecay        = 1    // -N коли все гаразд (поступове забування)
	stuckLimit        = 24   // поріг лічильника → вмикається фрустрація
	frustrationFrames = 24   // скільки кадрів напрямленого виходу
	stuckNearWall     = 0.5  // whisker ≥ цього = «біля стіни»
	stuckProgressEps  = 0.05 // наближення менше за це = «нема прогресу»
)

// ІНДЕКСИ СЛОТІВ у кадрі стану. Іменовані, а не «магічні числа», бо різні типи
// мозку кладуть у той самий кадр РІЗНІ дані, і зміна baseInputs інакше тихо
// ламає читання (напр. колись visible читався як cur[baseInputs-1] → зʼїхав би).
//
//	ПЕРЕСЛІДУВАЧ (GatherInputs)          ВБИВЦЯ (GatherKillerInputs)
//	[0,1]  напрямок до гравця            [0,1]  напрямок FLOW-FIELD (крізь стіни)
//	[2]    відстань по прямій            [2]    відстань ПО ЛАБІРИНТУ
//	[3,4]  швидкість ГРАВЦЯ              [3,4]  ВЛАСНА швидкість
//	[5..12] 8 whiskers                   [5..12] 8 whiskers
//	[13]   чи видно гравця (POMDP)       [13,14] швидкість цілі
//	[14,15] не вживаються (0)            [15]   власне HP
const (
	inDirX     = 0  // [0,1] напрямок до цілі: прямий (рій) або flow-field (вбивця)
	inDist     = 2  // відстань до цілі: по прямій (рій) або по лабіринту (вбивця)
	inVelX     = 3  // [3,4] швидкість: ГРАВЦЯ (рій) або ВЛАСНА (вбивця)
	inWhisker0 = 5  // [5..12] 8 променів-вусів; whisker[i] ↔ dirs8[i] ↔ дія i
	inVisible  = 13 // РІЙ: 1 = гравця видно (POMDP)
	inTgtVelX  = 13 // ВБИВЦЯ: [13,14] швидкість цілі (вести на випередження)
	inOwnHP    = 15 // ВБИВЦЯ: власне HP/MaxHP (коли відступати)
)

// [ПАМʼЯТЬ] Параметри стеку — РАНТАЙМНІ, на відміну від stackFrames.
//
// stackFrames лишається константою, бо задає розмір масиву входу мережі. Але
// ГЛИБИНА памʼяті (скільки з цих слотів реально несуть історію) і КРОК між
// семплами — звичайні змінні. Це дає порівняння «1 кадр проти 4» на мережі
// ОДНАКОВОГО розміру: змінюється лише інформація на вході, а не кількість ваг.
// Раніше ці два конфіги вимагали перекомпіляції з різним brainInputs, тож
// відрізнялись ще й місткістю мережі — зайва змінна в експерименті.
//
// memFrames = 1 → історія вимкнена, слоти 1..3 подаються нулями.
var (
	memFrames = stackFrames // скільки слотів стеку несуть історію (1..stackFrames)
	stackSkip = 10          // кадрів між семплами → вікно ≈ (memFrames-1)×stackSkip

	// [POMDP] Радіус видимості цілі (px) у режимі localSight.
	//
	// Var, бо саме він визначає СТЕЛЮ КОРИСНОСТІ ПАМʼЯТІ. Сліпий кадр у нас
	// порожній: слоти [0..4] і [13] обнуляються. Отже памʼять несе інформацію лише
	// тоді, коли запамʼятаний кадр був ЗРЯЧИМ. При видимості ~10% (виміряно: рій
	// наосліп 87–93% часу) памʼять корисна приблизно в одному кадрі з десяти —
	// звідси її стеля ~3 пункти, і звідси ж відсутність структури у свіпі вікна:
	// кількість інформативних кадрів задає видимість, а не довжина вікна.
	sightRange = float32(260.0)

	// [ГОРИЗОНТ] discount: наскільки цінувати майбутні нагороди (0..1).
	//
	// Ефективний горизонт ≈ 1/(1−γ) КРОКІВ, а крок у нас — один кадр при 120 TPS:
	//   γ=0.95  → 20 кроків  = 0.17с   ← довго був дефолтом
	//   γ=0.99  → 100 кроків = 0.83с
	//   γ=0.995 → 200 кроків = 1.67с
	//
	// Помічено при аналізі замірів памʼяті: найкорисніше вікно памʼяті — 1.5с, а
	// горизонт планування був 0.17с. Тобто агент мав на вході півтори секунди
	// минулого, але не міг ОЦІНИТИ переслідування, довше за одну шосту секунди.
	// Памʼять без відповідного горизонту нічого не варта — вона є, але політика не
	// має чим за неї заплатити. Var, щоб стенд перевірив цю пару разом.
	qGamma = float32(0.95)

	// Стеля Беллман-цілі (захист від розбіжності). ПІДБИРАТИ ЗА ФОРМУЛОЮ:
	// рівноважна цінність ≈ r/(1−γ). Виміряно: вбивця тримає r ≈ +0.74 → природна
	// Q ≈ 14.8 при γ=0.95, а стара стеля 10 її ОБРІЗАЛА в нормальному режимі, і
	// мережа переставала розрізняти «добре» і «дуже добре».
	//
	// УВАГА: стеля ЗАЛЕЖИТЬ від γ. Піднімаючи γ, треба піднімати й qClip у стільки
	// ж разів — інакше довший горизонт просто вріжеться в кліп і зміни не буде
	// видно (це та сама помилка, що вже раз коштувала нам вбивці).
	qClip = float32(25.0)

	// [RNN] Темп рекурентного навчання — рантаймний, бо ним перевіряли гіпотезу
	// недонавченості GRU. Нижчий за qLearnRate (0.005), бо BPTT дрейфував і maxQ
	// повзла до стелі.
	//
	// ГІПОТЕЗУ ВІДКИНУТО. GRU тримав maxQ ≈ 0 при 1.25–1.67 у стека, і здавалось,
	// що винне гальмування. Але на стенді (24 прогони на темп) 0.002 і 0.005
	// нерозрізненні (домінування 58%, p=0.32), і обидва нерозрізненні від конфігу
	// БЕЗ ПАМʼЯТІ взагалі. Причина не тут.
	//
	// Найімовірніша справжня: seqLen=8 кадрів розгортки BPTT — це 0.067с при 120
	// TPS, тоді як задачі потрібен горизонт ~1.5с (180 кадрів). Градієнт не дістає
	// туди, де лежить корисна інформація. Стек ті 1.5с має задарма — це вхідна фіча.
	gruLearnRate = float32(0.002)
	gruGradClip  = float32(0.5) // кліп градієнта по часу — тугіший = спокійніший BPTT
)

// sqrt2inv = 1/√2 — для діагональних напрямків (щоб були одиничної довжини).
const sqrt2inv = 0.70710678

// dirs8 — 8 напрямків руху за годинниковою (екранні координати: y росте ВНИЗ).
// Whisker[i] вимірює стіну в напрямку dirs8[i]; action[i] рухає туди ж —
// тобто мережа «бачить», наскільки заблокований кожен напрямок, у який може піти.
//
//	0=N(вгору) 1=NE 2=E(вправо) 3=SE 4=S(вниз) 5=SW 6=W(вліво) 7=NW
var dirs8 = [brainActions][2]float32{
	{0, -1}, {sqrt2inv, -sqrt2inv}, {1, 0}, {sqrt2inv, sqrt2inv},
	{0, 1}, {-sqrt2inv, sqrt2inv}, {-1, 0}, {-sqrt2inv, -sqrt2inv},
}

// Net — НЕЙРОМЕРЕЖА Q-агента: ваги + target-копія + буфер досвіду.
//
// Кілька Brain можуть указувати на ОДИН Net (режим sharedBrain=true) → «вулик-
// розум»: усі ділять одну вивчену політику й спільний досвід.
//
// [GO: ВАГИ ТРЬОХ ШАРІВ]
// W1: вхід → hidden1;  W2: hidden1 → hidden2;  W3: hidden2 → вихід(Q).
type Net struct {
	W1 [brainHidden1][brainInputs]float32
	B1 [brainHidden1]float32
	W2 [brainHidden2][brainHidden1]float32
	B2 [brainHidden2]float32
	W3 [brainActions][brainHidden2]float32
	B3 [brainActions]float32

	// [DQN: TARGET NETWORK] заморожена копія для Беллман-цілі (щоб не «тікала»).
	tW1         [brainHidden1][brainInputs]float32
	tB1         [brainHidden1]float32
	tW2         [brainHidden2][brainHidden1]float32
	tB2         [brainHidden2]float32
	tW3         [brainActions][brainHidden2]float32
	tB3         [brainActions]float32
	syncCounter int

	// [DQN: EXPERIENCE REPLAY] кільцевий буфер переходів (frame-stacking шлях).
	replay     []transition
	replayHead int
	replayFull bool

	// [RNN] Кільцевий буфер ВІДРІЗКІВ для рекурентного навчання (GRU шлях).
	// Той самий mu стереже обидва буфери (пишуть з паралельної фази).
	seqReplay []sequence
	seqHead   int
	seqFull   bool

	// [GO: MUTEX] захищає СПІЛЬНИЙ буфер від одночасного запису з різних горутин
	// (remember у паралельній фазі calcAcceleration). Ваги ж безпечні без локу
	// через РОЗДІЛЕННЯ ФАЗ: forward читається паралельно, train пише однопотоково
	// (g.trainBrains) — фази не перетинаються.
	mu sync.Mutex

	// [БАГАТО МОЗКІВ] Куди зберігати ці ваги. Різні ТИПИ ворогів мають різні
	// мережі й різні файли (рій → brainFile, вбивці → killerBrainFile).
	// Порожній рядок = ефемерна мережа, не зберігається (напр. мозок-жертва).
	file string

	// [МЕТРИКИ] акумулятори за період (скидаються в Metrics.collect). Пишуться
	// у tdUpdate (однопотоково в trainBrains) → без локу.
	mTDSum float32 // сума |TD-error| (сирого) — «здивування» мережі
	mQSum  float32 // сума max Q(s) — канарка розбіжності (росте безмежно = біда)
	mTDN   int     // кількість оновлень за період

	// [RNN/GRU] Ваги рекурентної клітини (вживаються лише коли useGRU=true).
	// GRU-клітина: вхід x(baseInputs) + попередній стан h(gruHidden) → новий h.
	//   z — update gate (скільки нового пускати в памʼять)
	//   r — reset gate (скільки старого забути перед оновленням)
	//   h~ — candidate (кандидат нового стану)
	// W* множать ВХІД, U* множать СТАН, B* — зсуви. Wq/Bq: стан h → Q(8) (лінійно).
	Wz [gruHidden][baseInputs]float32
	Uz [gruHidden][gruHidden]float32
	Bz [gruHidden]float32
	Wr [gruHidden][baseInputs]float32
	Ur [gruHidden][gruHidden]float32
	Br [gruHidden]float32
	Wh [gruHidden][baseInputs]float32
	Uh [gruHidden][gruHidden]float32
	Bh [gruHidden]float32
	Wq [brainActions][gruHidden]float32
	Bq [brainActions]float32

	// [RNN] Target-копії GRU-ваг — заморожені для Беллман-цілі (як tW1… для стеку).
	tWz [gruHidden][baseInputs]float32
	tUz [gruHidden][gruHidden]float32
	tBz [gruHidden]float32
	tWr [gruHidden][baseInputs]float32
	tUr [gruHidden][gruHidden]float32
	tBr [gruHidden]float32
	tWh [gruHidden][baseInputs]float32
	tUh [gruHidden][gruHidden]float32
	tBh [gruHidden]float32
	tWq [brainActions][gruHidden]float32
	tBq [brainActions]float32
}

// Brain — «голова» одного ворога-учня: указник на мережу + ОСОБИСТА пам'ять.
// Мережа може бути спільною; пам'ять (стан у часі, лічильники) — завжди своя,
// тож агенти діють індивідуально, але вчаться в (можливо) спільну мережу.
type Brain struct {
	net *Net

	// [RL: ПАМ'ЯТЬ МІЖ КАДРАМИ] — у кожного агента своя.
	// Reward за дію відомий лише НАСТУПНОГО кадру (коли побачимо результат руху).
	prevState  [brainInputs]float32 // попередній СТЕКНУТИЙ стан (для переходу)
	prevAction int
	hasPrev    bool

	// [ПАМ'ЯТЬ] Історія кадрів для frame-stacking: семпли кожні stackSkip кадрів.
	// Повний вхід = [поточний кадр | frames[0] | frames[1] | ...].
	frames    [stackFrames - 1][baseInputs]float32
	frameTick int

	// [RNN/GRU] Рекурентний прихований стан цього агента (вживається при useGRU).
	// Несеться між кадрами, скидається на новий епізод/respawn. Памʼять — своя в
	// кожного агента (як frames); ваги GRU — спільні в Net (вулик лишається).
	h [gruHidden]float32

	// [RNN крок 2] Накопичувач поточного відрізка траєкторії. Коли набереться
	// seqLen завершених кроків — відрізок їде в Net.seqReplay, лічильник у 0.
	gruPrevX [baseInputs]float32 // попередній вхідний кадр (для запису кроку)
	seqX     [seqTotal][baseInputs]float32
	seqA     [seqTotal]int
	seqR     [seqTotal]float32
	seqN     int

	age int // [3] вік (к-сть Step) — для автоспаду ε

	// [2] anti-stuck: лічильник застрягання, залишок кадрів «фрустрації», прапорець сліду.
	stuckCounter int
	frustration  int
	markStuck    bool

	// [SELF-PLAY] flee=true → ЖЕРТВА: reward інвертується (далі від ворога = краще).
	// false → ХИЖАК (ближче до гравця = краще), як у ворогів.
	flee bool

	// [БІЙ] combat=true → до нагороди додаються бойові члени (шкода/вбивства).
	// Лічильники наповнює resolveImpacts (кінець кадру), а споживає й обнуляє
	// rewardFor наступного кадру — той самий патерн «сигнал через кадр», що й
	// HitWall. Пишуться однопотоково → без гонок.
	combat   bool
	dmgDealt int // завдано шкоди від минулого Step
	dmgTaken int // отримано шкоди
	kills    int // добито цілей

	// [ВБИВЦЯ] flowNav=true → агент навігує за flow-field (а не по прямій). Впливає
	// на те, ЯКИЙ напрямок вважається «правильним», і на метрику blind-chase
	// (вбивця всевидющий, тож у неї не входить).
	flowNav bool

	// [RL: ВЛАСНИЙ ВНЕСОК] Прогрес агента за минулий кадр — проєкція ЙОГО ВЛАСНОЇ
	// швидкості на напрямок, куди йому треба. Заповнює Gather-функція (кожна знає
	// свій «правильний» напрямок), споживає rewardFor.
	//
	// Чому не «зміна відстані до цілі», як було спочатку: та включає рух САМОЇ ЦІЛІ,
	// якого агент не контролює. Виміряно на стенді: гравець (5.0 px/кадр) давав
	// коливання нагороди ±2.1, тоді як стеля власного внеску рою (1.2 px/кадр) —
	// лише ±0.6. Тобто ~75% сигналу було чужим рухом, і поведінка агента
	// перевертались залежно від того, тікав гравець чи налітав.
	progress float32

	// Для візуалізації (читає Draw, пише calcAcceleration — різні фази, без гонки).
	lastWhiskers [brainWhiskers]float32

	// [ФОРМА] Q-значення останнього рішення — для деформації тіла (див. updateBody).
	// Пишеться у Step, тобто в паралельній фазі, але у ВЛАСНИЙ Brain агента, як і
	// lastWhiskers → гонки немає. Читається однопотоково при малюванні.
	lastQ      [brainActions]float32
	lastAction int
	lastReward float32 // [МЕТРИКИ] нагорода останнього кроку (для середнього по рою)

	// [МЕТРИКИ ПАМʼЯТІ] Blind-pursuit: чи бачив агент гравця, КОЛИ обирав минулу
	// дію (visible на момент рішення). Дозволяє поміряти: коли агент СЛІПИЙ, чи
	// продовжує він скорочувати дистанцію (памʼять) чи блукає (реактивний амнезик).
	// Лічильники пише лише власна горутина агента (paralel-фаза) → без гонок;
	// collect() підсумовує їх однопотоково й скидає (як lastReward).
	prevVisible  bool
	mBlindN      int // «сліпих рішень» за період
	mBlindClosed int // ...із них скоротили дистанцію до гравця
}

// tanh — активація прихованого шару. Похідна: tanh'(z) = 1 - tanh(z)².
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// sigmoid — активація воріт GRU, стискає в (0,1) = «скільки пропустити».
// Похідна: σ'(z) = σ(z)·(1−σ(z)) — знадобиться для BPTT (крок 3).
func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

// NewNet створює мережу з Xavier-ініціалізацією (масштаб ~1/√fan_in),
// щоб tanh не входив у насичення і градієнт не зникав.
func NewNet() *Net {
	n := &Net{}
	s1 := float32(math.Sqrt(1.0 / brainInputs))
	for j := range n.W1 {
		for i := range n.W1[j] {
			n.W1[j][i] = (rand.Float32()*2 - 1) * s1
		}
	}
	s2 := float32(math.Sqrt(1.0 / brainHidden1))
	for k := range n.W2 {
		for j := range n.W2[k] {
			n.W2[k][j] = (rand.Float32()*2 - 1) * s2
		}
	}
	s3 := float32(math.Sqrt(1.0 / brainHidden2))
	for a := range n.W3 {
		for k := range n.W3[a] {
			n.W3[a][k] = (rand.Float32()*2 - 1) * s3
		}
	}
	n.initGRU()    // [RNN] ініціалізуємо й рекурентні ваги (навіть якщо useGRU=false)
	n.syncTarget() // target стартує копією живих ваг
	return n
}

// NewBrain — голова агента з ВЛАСНОЮ новою мережею (незалежний режим).
func NewBrain() *Brain { return &Brain{net: NewNet()} }

// NewBrainWith — голова агента, що ДІЛИТЬ передану мережу (режим sharedBrain).
func NewBrainWith(net *Net) *Brain { return &Brain{net: net} }

// syncTarget копіює живі ваги в target-мережу.
// [GO: масиви — значимі типи] присвоєння масиву копіює його повністю.
func (n *Net) syncTarget() {
	n.tW1, n.tB1 = n.W1, n.B1
	n.tW2, n.tB2 = n.W2, n.B2
	n.tW3, n.tB3 = n.W3, n.B3
	// [RNN] і GRU-ваги
	n.tWz, n.tUz, n.tBz = n.Wz, n.Uz, n.Bz
	n.tWr, n.tUr, n.tBr = n.Wr, n.Ur, n.Br
	n.tWh, n.tUh, n.tBh = n.Wh, n.Uh, n.Bh
	n.tWq, n.tBq = n.Wq, n.Bq
}

// argmaxQ повертає індекс дії з найбільшим Q.
func argmaxQ(q [brainActions]float32) int {
	best := 0
	for k := 1; k < brainActions; k++ {
		if q[k] > q[best] {
			best = k
		}
	}
	return best
}

// maxWhisker — найбільша близькість стіни серед 8 променів (0 = чисто навкруги).
// Читає вуса ПОТОЧНОГО кадру (індекси 5..12) — вони однакові і в стеку, і в GRU.
func maxWhisker(f [baseInputs]float32) float32 {
	m := f[inWhisker0]
	for i := 1; i < brainWhiskers; i++ {
		if f[inWhisker0+i] > m {
			m = f[inWhisker0+i]
		}
	}
	return m
}

// escapeAction — напрямок із НАЙМЕНШОЮ близькістю стіни (найвідкритіший), щоб
// гарантовано вийти з пастки, а не смикатись на місці. Серед однаково відкритих
// напрямків — рівноймовірно (reservoir), аби агенти не злипались в один бік.
func escapeAction(f [baseInputs]float32) int {
	best, bestW, ties := 0, f[inWhisker0], 1
	for i := 1; i < brainWhiskers; i++ {
		w := f[inWhisker0+i]
		switch {
		case w < bestW:
			best, bestW, ties = i, w, 1
		case w == bestW:
			ties++
			if rand.Intn(ties) == 0 {
				best = i
			}
		}
	}
	return best
}

// epsilon — поточна ε з лінійним автоспадом max→min за qEpsilonDecay кроків,
// але не нижче floor (qEpsilonMin). Свіжий мозок (age=0) досліджує найбільше,
// натренований (age велике / завантажений) — майже чистий мисливець.
func (b *Brain) epsilon() float32 {
	if frozenPolicy {
		return 0 // [ЗАМІР] жодних випадкових дій → політика детермінована
	}
	if !epsilonDecayEnabled {
		return qEpsilonConst // спад вимкнено → ε постійна, не змінюється
	}
	t := float32(b.age) / qEpsilonDecay
	if t > 1 {
		t = 1
	}
	return qEpsilonMax + (qEpsilonMin-qEpsilonMax)*t
}

// selectAction — ε-GREEDY вибір дії (читає ваги мережі).
//
// [RL: EXPLORATION vs EXPLOITATION]
// З імовірністю ε — випадкова дія (досліджуємо світ). Інакше — найкраща за Q.
func (b *Brain) selectAction(state [brainInputs]float32) int {
	if rand.Float32() < b.epsilon() {
		return rand.Intn(brainActions)
	}
	q, _, _ := b.net.forwardQ(state)
	return argmaxQ(q)
}

// selectFromQ — ε-greedy на ВЖЕ обрахованому q-векторі (шлях GRU: q дає forwardGRU,
// повторно рахувати не треба). Та сама логіка, що й selectAction.
func (b *Brain) selectFromQ(q [brainActions]float32) int {
	if rand.Float32() < b.epsilon() {
		return rand.Intn(brainActions)
	}
	return argmaxQ(q)
}

// progressToward — «наскільки агент САМ просунувся туди, куди йому треба».
// Одна величина живить і нагороду, і anti-stuck, тож вони не суперечать.
//
// Правило одне для всіх типів мозку — різниться лише НАПРЯМОК, який Gather-функція
// вважає правильним:
//
//	РІЙ / ЖЕРТВА  → напрямок ПО ПРЯМІЙ до цілі (вони не знають лабіринту)
//	ВБИВЦЯ        → напрямок FLOW-FIELD, тобто вздовж коридору крізь стіни
//
// [ЧОМУ НЕ «зміна відстані»] Спочатку прогрес рахувався як (prevDist − dist).
// Розкладемо його:
//
//	prevDist − dist ≈ dot(ВЛАСНА швидкість, напрямок) − dot(швидкість ЦІЛІ, напрямок)
//	                  └─ те, що агент контролює ─┘      └─ для нього чистий шум ─┘
//
// Другий доданок ми викинули. На стенді він давав ~75% розмаху нагороди (гравець
// швидший за рій у 4 рази), через що сигнал/шум був ~1:50, а поведінка агента
// перевертались залежно від стилю гравця: тікаєш — вчиться наздоганяти, налітаєш —
// вчиться відступати (бо дистанція скорочується й без його зусиль).
//
// [ЧОМУ НЕ «відстань по лабіринту» для вбивці] Та рахується в КЛІТИНКАХ і
// міняється раз на ~20 кадрів → нагорода йшла б рваними стрибками. Проєкція
// швидкості гладка, бо швидкість неперервна.
func (b *Brain) progressToward() float32 { return b.progress }

// rewardFor — [RL: REWARD SHAPING] СПІЛЬНА нагорода за минулу дію для обох
// шляхів памʼяті (раніше цей код був продубльований у stepStack і stepGRU).
//
//	prevWhisker — близькість стіни в НАПРЯМКУ, куди агент пішов минулого кадру
//	              (беремо з його ж попереднього кадру: стек чи gruPrevX).
//
// Складові:
//   - щільна: наближення до цілі (жертва з flee=true — навпаки, віддалення);
//   - штрафи за стіни: по факту удару + плавний градієнт «тримай дистанцію»;
//   - [БІЙ] рідкісні бойові події, лише якщо combat=true.
//
// Лічильники шкоди обнуляються ЗАВЖДИ — інакше в не-бойових мозків вони росли б
// вічно й вистрілили б, якби combat колись увімкнули.
func (b *Brain) rewardFor(hitWall bool, prevWhisker float32) float32 {
	sign := float32(1)
	if b.flee {
		sign = -1
	}
	r := sign * b.progressToward() * rewardCloserScale
	if hitWall {
		r += rewardWallHit
	}
	r += rewardNearWall * prevWhisker

	if b.combat {
		r += rewardDamageDealt * float32(b.dmgDealt)
		r += rewardDamageTaken * float32(b.dmgTaken)
		r += rewardKill * float32(b.kills)
	}
	b.dmgDealt, b.dmgTaken, b.kills = 0, 0, 0

	b.lastReward = r // [МЕТРИКИ] для середньої нагороди по рою
	return r
}

// Step — один крок агента: оцінити минулу дію й обрати нову. Викликається
// щокадру в паралельній фазі (calcAcceleration). НЕ тренує мережу — лише кладе
// досвід у буфер; навчання відбувається раз/кадр однопотоково в g.trainBrains()
// (мережа може бути спільною: тренувати її N×qBatch/кадр було б і неправильно,
// і небезпечно для гонок).
//
// [ДИСПЕТЧЕР] Тут обирається ШЛЯХ ПАМʼЯТІ: рекурентний GRU (brain_gru.go) або
// frame-stacking (brain_stack.go). Усе інше в них — незалежне.
func (b *Brain) Step(cur [baseInputs]float32, hitWall bool) int {
	b.age++ // [3] для автоспаду ε
	if useGRU {
		return b.stepGRU(cur, hitWall)
	}
	return b.stepStack(cur, hitWall)
}

// train — k оновлень на випадкових вибірках із буфера (серце DQN).
//
// Викликається ОДИН раз за кадр ОДНОПОТОКОВО (g.trainBrains, поза паралельною
// фазою) → запис ваг безпечний без локу. Зі спільним мозком уся колективна
// вибірка тренує ОДНУ мережу нормальним темпом (а не N×qBatch разів за кадр).
//
// [ДИСПЕТЧЕР] Той самий поділ шляхів, що й у Step.
func (n *Net) train(k int) {
	if useGRU {
		n.trainSeq(seqBatch) // [RNN] рекурентний шлях — навчання на відрізках (BPTT)
		return
	}
	m := n.replayLen()
	if m < qMinReplay {
		return
	}
	for i := 0; i < k; i++ {
		t := n.replay[rand.Intn(m)]
		n.tdUpdate(t.s, t.a, t.r, t.s2)
	}
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// wallWhisker — промінь-сенсор: відстань від точки (cx,cy) до стіни в напрямку (dx,dy).
//
// [AI: RAYCAST]
// Крокуємо вздовж напрямку, поки не натрапимо на стіну (або не вийдемо за range).
// Повертаємо БЛИЗЬКІСТЬ: 1 = стіна впритул, 0 = чисто на всю довжину.
func wallWhisker(cx, cy, dx, dy float32) float32 {
	for d := float32(whiskerStep); d <= whiskerRange; d += whiskerStep {
		px := cx + dx*d
		py := cy + dy*d
		if isWallAt(int(px)/pixelSize, int(py)/pixelSize) {
			return 1 - d/whiskerRange
		}
	}
	return 0
}

// hasLineOfSight — чи є пряма видимість між точками (немає стіни на прямій).
// Крокуємо від (x1,y1) до (x2,y2) по пів-клітинки; якщо натрапили на стіну до
// цілі — видимості нема. [POMDP] так гравець «ховається» за стінами.
func hasLineOfSight(x1, y1, x2, y2 float32) bool {
	dx := x2 - x1
	dy := y2 - y1
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	steps := int(dist / (pixelSize / 2))
	if steps < 1 {
		return true
	}
	sx := dx / float32(steps)
	sy := dy / float32(steps)
	x, y := x1, y1
	for i := 0; i < steps; i++ {
		x += sx
		y += sy
		if isWallAt(int(x)/pixelSize, int(y)/pixelSize) {
			return false
		}
	}
	return true
}

// GatherInputs збирає стан (state) для Q-мережі.
//
//	[0] dx/screenWidth     напрямок до гравця X   ◄─┐
//	[1] dy/screenHeight    напрямок до гравця Y     │ обнуляються,
//	[2] dist/screenWidth   відстань до гравця       │ коли гравця НЕ видно
//	[3] pVelX/5            швидкість гравця X        │ (POMDP)
//	[4] pVelY/5            швидкість гравця Y     ◄─┘
//	[5..12] whiskers       близькість стіни у 8 напрямках  ◄── зір на перешкоди (завжди)
//	[13] visible           1 = гравця видно, 0 = ні
//
// [POMDP] Якщо localSight — гравець «видимий» лише в межах sightRange і по прямій
// видимості (промінь не перекритий стіною). Поза цим позиційні входи = 0 і
// visible = 0 → агент не знає, де гравець. Саме тут згодом порятує пам'ять.
//
// Побічно зберігає whiskers у Brain для візуалізації.
// Повертає ОДИН кадр (baseInputs); склеювання в стек робить Brain.Step.
func GatherInputs(enemy, player *Pixel) [baseInputs]float32 {
	dx := player.X - enemy.X
	dy := player.Y - enemy.Y
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if dist == 0 {
		dist = 1
	}

	cx := enemy.X + pixelSize/2
	cy := enemy.Y + pixelSize/2

	visible := true
	if localSight {
		visible = dist <= sightRange &&
			hasLineOfSight(cx, cy, player.X+pixelSize/2, player.Y+pixelSize/2)
	}

	var in [baseInputs]float32
	if visible {
		in[0] = dx / screenWidth
		in[1] = dy / screenHeight
		in[2] = dist / screenWidth
		in[3] = player.VelX / 5.0
		in[4] = player.VelY / 5.0
		in[inVisible] = 1
	}
	// (якщо не видно — [0..4] і [13] лишаються 0)

	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[inWhisker0+i] = w
		if enemy.Brain != nil {
			enemy.Brain.lastWhiskers[i] = w
		}
	}

	// [RL] Прогрес за минулий кадр — ВЛАСНИЙ рух агента в бік цілі.
	// Рахуємо від ІСТИННОГО напрямку, навіть коли ціль не видно: нагороду видає
	// середовище, а не сприйняття агента — інакше сліпі кадри лишились би зовсім
	// без сигналу. (Швидкість тут «вчорашня»: Step іде до updateUnits, тобто це
	// саме те переміщення, яке щойно відбулось.)
	if enemy.Brain != nil {
		enemy.Brain.progress = closingSpeed(enemy.VelX, enemy.VelY, dx/dist, dy/dist)
	}
	return in
}

// nearestHostile — [КОМАНДИ] найближчий живий юніт ЧУЖОЇ фракції (або nil).
//
// Ключова функція командного бою: ціль більше не «завжди гравець». Свої одне
// одного ігнорують — інакше юніти гравця тікали б від власних союзників, а рій
// бив би своїх.
func nearestHostile(from *Pixel, units []Pixel) *Pixel {
	var best *Pixel
	var bestD float32 = 1e30
	for i := range units {
		u := &units[i]
		if u == from || u.HP <= 0 || u.Faction == from.Faction {
			continue
		}
		dx, dy := u.X-from.X, u.Y-from.Y
		if d := dx*dx + dy*dy; d < bestD {
			bestD, best = d, u
		}
	}
	return best
}

// GatherPreyInputs — стан для мозку-ЖЕРТВИ (гравця у self-play). Дзеркало
// GatherInputs: замість «куди гравець» — «звідки загроза» (напрямок до НАЙБЛИЖЧОГО
// ворога). Мережа вчиться рухатись ГЕТЬ (бо reward інвертований, flee=true).
// Повертає кадр (baseInputs) і відстань до найближчого ворога (для reward).
//
// [КОМАНДИ] Ціль шукаємо через nearestHostile: відколи на полі є юніти гравця,
// «найближчий юніт» ≠ «найближча загроза» — від своїх тікати не треба.
func GatherPreyInputs(player *Pixel, units []Pixel) [baseInputs]float32 {
	cx := player.X + pixelSize/2
	cy := player.Y + pixelSize/2

	var in [baseInputs]float32
	if e := nearestHostile(player, units); e != nil {
		dx := e.X - player.X
		dy := e.Y - player.Y
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if dist == 0 {
			dist = 1
		}
		in[0] = dx / screenWidth
		in[1] = dy / screenHeight
		in[2] = dist / screenWidth
		in[3] = e.VelX / 5.0
		in[4] = e.VelY / 5.0
		in[inVisible] = 1
	}

	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[inWhisker0+i] = w
		if player.Brain != nil {
			player.Brain.lastWhiskers[i] = w
		}
	}

	// [RL] Прогрес — власний рух У БІК загрози. Для жертви (flee=true) знак
	// нагороди інвертується, тож рух ГЕТЬ від загрози й дає плюс.
	if player.Brain != nil {
		if e := nearestHostile(player, units); e != nil {
			dx, dy := e.X-player.X, e.Y-player.Y
			if d := float32(math.Sqrt(float64(dx*dx + dy*dy))); d > 0 {
				player.Brain.progress = closingSpeed(player.VelX, player.VelY, dx/d, dy/d)
			}
		} else {
			player.Brain.progress = 0
		}
	}
	return in
}

// GatherKillerInputs — [ВБИВЦЯ] стан для мозку, що ЗНАЄ ЛАБІРИНТ.
//
// Той самий РОЗМІР кадру, що й у рою, але інший СЕНС слотів (див. таблицю вгорі).
// Ключова різниця: замість «куди гравець по прямій» — «куди ЙТИ лабіринтом»
// (flow-field). Мережа не витрачає ємність на розвʼязання лабіринту — навігація
// їй підказана, і вся ємність іде на ТАКТИКУ: коли кинутись, коли відступити.
//
// Це свідомий поділ праці: те, що добре рахує алгоритм (BFS), не варто вчити
// градієнтним спуском. Добрі фічі сильніші за більшу мережу.
//
// [GO: ПАРАЛЕЛЬНЕ ЧИТАННЯ] flow читається з горутин воркер-пулу, але поле
// перебудовується РАНІШЕ в Update (однопотоково) → та сама «розділення фаз»,
// що й для boidMap. Гонок нема.
func GatherKillerInputs(enemy, player *Pixel, flow *FlowField) [baseInputs]float32 {
	cx := enemy.X + pixelSize/2
	cy := enemy.Y + pixelSize/2

	var in [baseInputs]float32

	// [0,1] напрямок КРІЗЬ СТІНИ + [2] відстань ПО ЛАБІРИНТУ.
	// Заразом рахуємо progress — рух УЗДОВЖ коридору за минулий кадр
	// (швидкість тут іще «вчорашня»: калькуляція йде до updateUnits, тобто це
	// саме те переміщення, яке щойно відбулось). Його споживає rewardFor.
	if dx, dy, ok := flow.dirAt(enemy.X, enemy.Y); ok {
		in[inDirX], in[inDirX+1] = dx, dy
		in[inDist] = clamp(float32(flow.distAt(enemy.X, enemy.Y))/flowDistNorm, 0, 1)
		if enemy.Brain != nil {
			enemy.Brain.progress = closingSpeed(enemy.VelX, enemy.VelY, dx, dy)
		}
	} else {
		in[inDist] = 1 // шляху нема (замкнена кишеня) → «нескінченно далеко»
		if enemy.Brain != nil {
			enemy.Brain.progress = 0
		}
	}

	// [3,4] ВЛАСНА швидкість, нормалізована власним максимумом → [-1..1].
	// Критично для «кидка кобри»: шкода залежить від швидкості зближення, тож
	// агент мусить ВІДЧУВАТИ, наскільки він розігнався. Вивести це з послідовності
	// [2] він не зміг би — відстань квантована клітинками (міняється раз на ~20 кадрів).
	if m := enemy.Cfg.MaxSpeed; m > 0 {
		in[inVelX] = clamp(enemy.VelX/m, -1, 1)
		in[inVelX+1] = clamp(enemy.VelY/m, -1, 1)
	}

	// [5..12] вуса — той самий зір на стіни, що й у рою (мікроманевр упритул).
	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[inWhisker0+i] = w
		if enemy.Brain != nil {
			enemy.Brain.lastWhiskers[i] = w
		}
	}

	// [13,14] швидкість цілі (вести на випередження) + [15] власне HP.
	in[inTgtVelX] = player.VelX / 5.0
	in[inTgtVelX+1] = player.VelY / 5.0
	if enemy.MaxHP > 0 {
		in[inOwnHP] = float32(enemy.HP) / float32(enemy.MaxHP)
	}
	return in
}

// Файли ваг — по одному на ТИП мозку (різні типи вчаться незалежно).
const (
	brainFile      = "brain_weights.json"       // ворожий рій-переслідувач (ConfigLearner)
	killerFile     = "killer_weights.json"      // ворог-вбивця з flow-field (ConfigKiller)
	allyFile       = "ally_weights.json"        // [КОМАНДИ] переслідувач гравця
	allyKillerFile = "ally_killer_weights.json" // [КОМАНДИ] вбивця гравця
)

// BrainData — серіалізація ваг у JSON + розміри мережі для перевірки сумісності.
type BrainData struct {
	Inputs  int `json:"inputs"`
	Hidden1 int `json:"hidden1"`
	Hidden2 int `json:"hidden2"`
	Actions int `json:"actions"`

	W1 [brainHidden1][brainInputs]float32  `json:"w1"`
	B1 [brainHidden1]float32               `json:"b1"`
	W2 [brainHidden2][brainHidden1]float32 `json:"w2"`
	B2 [brainHidden2]float32               `json:"b2"`
	W3 [brainActions][brainHidden2]float32 `json:"w3"`
	B3 [brainActions]float32               `json:"b3"`

	// [RNN] Ваги GRU. Старі файли їх не містять (HasGRU=false) → GRU стартує з нуля
	// через initGRU. GruHidden звіряємо окремо (розмір h) при завантаженні.
	HasGRU    bool                             `json:"has_gru"`
	GruHidden int                              `json:"gru_hidden"`
	Wz        [gruHidden][baseInputs]float32   `json:"wz"`
	Uz        [gruHidden][gruHidden]float32    `json:"uz"`
	Bz        [gruHidden]float32               `json:"bz"`
	Wr        [gruHidden][baseInputs]float32   `json:"wr"`
	Ur        [gruHidden][gruHidden]float32    `json:"ur"`
	Br        [gruHidden]float32               `json:"br"`
	Wh        [gruHidden][baseInputs]float32   `json:"wh"`
	Uh        [gruHidden][gruHidden]float32    `json:"uh"`
	Bh        [gruHidden]float32               `json:"bh"`
	Wq        [brainActions][gruHidden]float32 `json:"wq"`
	Bq        [brainActions]float32            `json:"bq"`
}

// SaveNet зберігає ваги у ВЛАСНИЙ файл мережі (n.file). Ефемерні мережі
// (file == "", напр. мозок-жертва в self-play) не зберігаються.
func SaveNet(n *Net) error {
	if n == nil || n.file == "" {
		return nil
	}
	return saveNetTo(n, n.file)
}

// newNetFor — нова або завантажена мережа для конкретного ТИПУ мозку.
// Повертає також loaded: чи ваги реально прийшли з файлу (навчена → ε на floor).
func newNetFor(path string) (n *Net, loaded bool) {
	if n = loadNetFrom(path); n != nil {
		return n, true
	}
	n = NewNet()
	n.file = path
	return n, false
}

// saveNetTo серіалізує мережу (стек + GRU ваги) у JSON за вказаним шляхом.
func saveNetTo(n *Net, path string) error {
	data := BrainData{
		Inputs: brainInputs, Hidden1: brainHidden1, Hidden2: brainHidden2, Actions: brainActions,
		W1: n.W1, B1: n.B1, W2: n.W2, B2: n.B2, W3: n.W3, B3: n.B3,
		// [RNN] і рекурентні ваги — щоб gru-рій не вчився з нуля щоразу.
		HasGRU: true, GruHidden: gruHidden,
		Wz: n.Wz, Uz: n.Uz, Bz: n.Bz,
		Wr: n.Wr, Ur: n.Ur, Br: n.Br,
		Wh: n.Wh, Uh: n.Uh, Bh: n.Bh,
		Wq: n.Wq, Bq: n.Bq,
	}
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}

// LoadNet завантажує мережу з файлу за замовчуванням (brainFile).
func LoadNet() *Net { return loadNetFrom(brainFile) }

// loadNetFrom завантажує мережу з файлу. Повертає nil (→ caller створить NewNet),
// якщо файлу немає, він пошкоджений, або РОЗМІРИ стек-мережі не збігаються.
// GRU-ваги вантажимо, ЛИШЕ якщо файл їх містить і розмір h збігається; інакше —
// initGRU (стара збірка чи інший gruHidden → рекурентна памʼять з нуля, стек цілий).
func loadNetFrom(path string) *Net {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var data BrainData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil
	}
	if data.Inputs != brainInputs || data.Hidden1 != brainHidden1 ||
		data.Hidden2 != brainHidden2 || data.Actions != brainActions {
		return nil // несумісна архітектура → почнемо з нуля
	}
	n := &Net{W1: data.W1, B1: data.B1, W2: data.W2, B2: data.B2, W3: data.W3, B3: data.B3}
	n.file = path // мережа памʼятає, звідки прийшла → туди ж і збережеться
	if data.HasGRU && data.GruHidden == gruHidden {
		n.Wz, n.Uz, n.Bz = data.Wz, data.Uz, data.Bz
		n.Wr, n.Ur, n.Br = data.Wr, data.Ur, data.Br
		n.Wh, n.Uh, n.Bh = data.Wh, data.Uh, data.Bh
		n.Wq, n.Bq = data.Wq, data.Bq
	} else {
		n.initGRU() // немає ваг GRU у файлі / інший розмір → рекурентна памʼять з нуля
	}
	n.syncTarget()
	return n
}
