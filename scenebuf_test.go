package main

import "testing"

// sandboxScene — зберегти й повернути весь стан буфера сцени.
//
// Стан тут пакетний (sceneBuf, sceneBufFor, sceneScale, renderScale, ssLadder), і
// лишити його зміненим означало б зіпсувати сусідні тести мовчки — рівно той клас
// протікання, від якого страхує sandboxSettings.
func sandboxScene(t *testing.T) {
	t.Helper()
	savedBuf, savedFor := sceneBuf, sceneBufFor
	savedScene, savedRender := sceneScale, renderScale
	savedLadder := ssLadder
	sceneBuf, sceneBufFor = nil, 0
	t.Cleanup(func() {
		releaseSceneBuf()
		sceneBuf, sceneBufFor = savedBuf, savedFor
		sceneScale, renderScale = savedScene, savedRender
		ssLadder = savedLadder
	})
}

// TestSceneBufIsReleasedWhenSupersamplingOff — спуск на 1× ВІДДАЄ памʼять.
//
// Рання гілка `renderScale <= 1` поверталась до того, як код доходив до звільнення,
// і правило «звільняємо СПЕРШУ: на верхніх щаблях буфер важить сотні мегабайтів»
// стояло рядків на десять нижче — тобто на цьому шляху не діяло.
//
// Сценарій — ОДНЕ натискання: ssLadder завертається 4× → 1× на панелі, і буфер
// 6800×3920 (≈107 МБ) лишався виділеним до кінця сесії. Помітити це в грі майже
// неможливо: картинка правильна, FPS правильний, просто памʼять не вертається.
func TestSceneBufIsReleasedWhenSupersamplingOff(t *testing.T) {
	sandboxScene(t)

	renderScale = 2
	if beginScene(); sceneBuf == nil {
		t.Fatal("буфер не створився при renderScale = 2 — перевіряти нічого")
	}

	renderScale = 1
	beginScene()
	if sceneBuf != nil {
		t.Error("після спуску на 1× буфер лишився висіти — це витік до кінця сесії")
	}
	if sceneBufFor != 0 {
		t.Errorf("sceneBufFor = %v після звільнення: кешовано рішення про буфер, "+
			"якого вже немає", sceneBufFor)
	}
	if sceneScale != 1 {
		t.Errorf("sceneScale = %v при вимкненому суперсемплінгу", sceneScale)
	}
}

// TestSceneBufCachesTheRequestNotTheGrantedSize — кеш ключується ЗАПИТОМ.
//
// Тут і був другий дефект, небезпечніший за витік. Кеш порівнював розміри буфера з
// розмірами, порахованими від renderScale — тобто від ЗАПИТУ. Доки драбина віддавала
// саме запитане, це збігалось. Але вона на те й драбина, щоб спускатись нижче: щойно
// видане ≠ запитаного, розміри не сходились уже НІКОЛИ, і кожен кадр робив Deallocate,
// повторний прохід драбини й повторну спробу того самого виділення, яке щойно
// провалилось. Тобто 60 разів на секунду повторювалось рівно те виділення, яке на ×12
// поклало процес ПОВЗ recover (див. ssLadder). Замість плавної деградації — найгірший
// можливий шлях, і саме тоді, коли залізу вже важко.
//
// [ЯК ЦЕ ЗМІРЯТИ БЕЗ ВІДМОВИ ВИДІЛЕННЯ] Відмову GPU в тесті не викликати, але вона й
// не потрібна: розбіжність «запит vs видане» дає сама ДРАБИНА. Підміняємо ssLadder на
// {1, 2} і просимо 4 — верхнього щабля просто немає, тож видається 2 при запиті 4.
// Це та сама розбіжність, тільки досяжна детерміновано.
func TestSceneBufCachesTheRequestNotTheGrantedSize(t *testing.T) {
	sandboxScene(t)

	ssLadder = []float32{1, 2}
	renderScale = 4 // щабля 4 в підміненій драбині немає → спуск до 2

	first := beginScene()
	if first == nil || sceneBuf == nil {
		t.Fatal("драбина не видала буфера навіть на щаблі 2")
	}
	if sceneScale != 2 {
		t.Fatalf("видано масштаб %v, очікували 2 — підміна драбини не спрацювала", sceneScale)
	}
	if sceneBufFor != 4 {
		t.Errorf("sceneBufFor = %v, очікували 4: памʼятати треба ЗАПИТ, інакше кеш "+
			"не збіжиться ніколи", sceneBufFor)
	}

	second := beginScene()
	if second != first {
		t.Error("буфер перебудовано на другому ж кадрі при незмінному запиті — драбина " +
			"крутиться щокадру, разом із повторною спробою щабля, що вже провалився")
	}
	if sceneScale != 2 {
		t.Errorf("sceneScale поїхав на %v при повторному виклику", sceneScale)
	}
}

// TestSceneBufDoesNotRetryAFailedLadderEveryFrame — «не влізло нічого» теж кешується.
//
// Окремий випадок того самого: якщо драбина не дала НІЧОГО, sceneBuf лишається nil.
// Кеш, який перевіряє лише `sceneBuf != nil`, у цьому стані промахувався б щокадру й
// повторював увесь спуск — тобто найдорожчу гілку в найгіршому стані заліза. Тому
// умовою кешу є sceneBufFor, а не наявність буфера.
//
// Драбина з самих одиниць не дає жодного щабля > 1, тож спуск гарантовано порожній.
func TestSceneBufDoesNotRetryAFailedLadderEveryFrame(t *testing.T) {
	sandboxScene(t)

	ssLadder = []float32{1}
	renderScale = 4

	beginScene()
	if sceneBuf != nil {
		t.Fatal("драбина з самих одиниць видала буфер — сцену тесту зламано")
	}
	if sceneBufFor != 4 {
		t.Errorf("sceneBufFor = %v після невдалого спуску, очікували 4: без цього "+
			"порожня драбина проходилась би щокадру", sceneBufFor)
	}
	if sceneScale != 1 {
		t.Errorf("sceneScale = %v, очікували 1 — камера множила б на масштаб, якого "+
			"немає, і світ поїхав би за межі екрана", sceneScale)
	}
}
