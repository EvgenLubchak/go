package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// sandboxSettings — тимчасовий settings.json + відкат глобалей після тесту.
func sandboxSettings(t *testing.T) {
	t.Helper()
	savedPath, savedAA, savedSS, savedTPS := settingsPath, antiAlias, renderScale, gameTPS
	settingsPath = filepath.Join(t.TempDir(), "settings.json")
	t.Cleanup(func() {
		settingsPath, antiAlias, renderScale, gameTPS = savedPath, savedAA, savedSS, savedTPS
	})
}

// TestSettingsMissingKeyKeepsDefault — PATCH-семантика: відсутній ключ означає
// «не задано → дефолт», а НЕ нульове значення. Користувач, що зберіг лише ss,
// не просив вимкнути aa.
func TestSettingsMissingKeyKeepsDefault(t *testing.T) {
	sandboxSettings(t)
	antiAlias, renderScale = antiAliasDefault, renderScaleDefault

	if err := os.WriteFile(settingsPath, []byte(`{"ss": 2}`), 0644); err != nil {
		t.Fatal(err)
	}
	loadSettings()
	if renderScale != 2 {
		t.Errorf("ss з файлу не застосувався: %g", renderScale)
	}
	if antiAlias != antiAliasDefault {
		t.Error("відсутній ключ aa перезаписав дефолт — PATCH-семантика зламана")
	}
}

// TestSettingsSavesOnlyDeviations — у файл ідуть ЛИШЕ відхилення від дефолтів.
// Інакше сьогоднішні дефолти замерзають у файлі користувача, і майбутня зміна
// дефолту в коді тихо не долітає (клас підмін «IDE-тека з вагами»).
func TestSettingsSavesOnlyDeviations(t *testing.T) {
	sandboxSettings(t)

	antiAlias, renderScale = antiAliasDefault, 2 // відхилення лише в SS
	saveSettings()
	raw, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"ss"`) {
		t.Error("відхилення ss не записалось")
	}
	if strings.Contains(string(raw), `"aa"`) {
		t.Errorf("дефолтне aa потрапило у файл — воно замерзне: %s", raw)
	}

	// Повернення до дефолту мусить ПРИБРАТИ ключ, а не записати дефолтне значення.
	renderScale = renderScaleDefault
	saveSettings()
	raw, _ = os.ReadFile(settingsPath)
	if strings.Contains(string(raw), `"ss"`) {
		t.Errorf("ss повернувся до дефолту, але ключ лишився у файлі: %s", raw)
	}
}

// TestSettingsRoundTrip — записане відхилення переживає цикл save → load.
func TestSettingsRoundTrip(t *testing.T) {
	sandboxSettings(t)

	antiAlias, renderScale = !antiAliasDefault, 4
	saveSettings()
	antiAlias, renderScale = antiAliasDefault, renderScaleDefault // «перезапуск»
	loadSettings()
	if antiAlias == antiAliasDefault || renderScale != 4 {
		t.Errorf("round-trip загубив відхилення: aa=%v ss=%g", antiAlias, renderScale)
	}
}

// TestSettingsInvalidSSClamped — файл це ВВІД КОРИСТУВАЧА, і в SS уже є шрам:
// ×12 валив гру повз recover (гігабайтний буфер убиває драйвер). Незнайоме
// значення клемпиться до найближчого ПЕРЕВІРЕНОГО щабля драбини, не застосовується.
func TestSettingsInvalidSSClamped(t *testing.T) {
	sandboxSettings(t)

	cases := []struct{ in, want float32 }{
		{12, 4},    // за стелею → верхній щабель
		{1.7, 1.5}, // між щаблями → найближчий
		{-3, 1},    // сміття → нижній щабель
		{2, 2},     // легальний — як є
	}
	for _, c := range cases {
		antiAlias, renderScale = antiAliasDefault, renderScaleDefault
		if err := os.WriteFile(settingsPath, []byte(
			`{"ss": `+trimFloat(c.in)+`}`), 0644); err != nil {
			t.Fatal(err)
		}
		loadSettings()
		if renderScale != c.want {
			t.Errorf("ss %g з файлу → %g, очікувалось %g", c.in, renderScale, c.want)
		}
	}
}

func trimFloat(v float32) string {
	return strconv.FormatFloat(float64(v), 'g', -1, 32)
}

// TestPanelActionsPersist — дії ПАНЕЛІ (та сама таблиця panelItems) реально
// міняють значення і пишуть файл. Стереже звʼязку «таблиця → рантайм → диск»:
// панель, що перемикає але не зберігає, виглядала б робочою до перезапуску.
func TestPanelActionsPersist(t *testing.T) {
	sandboxSettings(t)
	antiAlias, renderScale = antiAliasDefault, renderScaleDefault

	// SS: крок уперед драбиною (1.5 → 2) і назад (→ 1.5).
	var ss *panelItem
	var aa *panelItem
	for i := range panelItems {
		if strings.HasPrefix(panelItems[i].name, "SS") {
			ss = &panelItems[i]
		}
		if strings.HasPrefix(panelItems[i].name, "AA") {
			aa = &panelItems[i]
		}
	}
	if ss == nil || aa == nil {
		t.Fatal("панель не має пунктів AA/SS — таблиця розійшлась із тестом")
	}

	ss.next(nil)
	if renderScale != 2 {
		t.Errorf("панельний крок SS: %g, очікувалось 2", renderScale)
	}
	raw, err := os.ReadFile(settingsPath)
	if err != nil || !strings.Contains(string(raw), `"ss"`) {
		t.Error("крок SS не зберігся у файл")
	}
	ss.prev(nil)
	if renderScale != renderScaleDefault {
		t.Errorf("панельний крок SS назад: %g", renderScale)
	}

	aa.next(nil)
	if antiAlias == antiAliasDefault {
		t.Error("панельний фліп AA не змінив значення")
	}
	raw, _ = os.ReadFile(settingsPath)
	if !strings.Contains(string(raw), `"aa"`) {
		t.Error("фліп AA не зберігся у файл")
	}

	// TPS: перемикач 60↔120 через таблицю (g=nil у тесті: біт не чіпається).
	var tps *panelItem
	for i := range panelItems {
		if strings.HasPrefix(panelItems[i].name, "TPS") {
			tps = &panelItems[i]
		}
	}
	if tps == nil {
		t.Fatal("панель не має пункту TPS")
	}
	gameTPS = gameTPSDefault
	tps.next(nil)
	if gameTPS == gameTPSDefault {
		t.Error("панельний фліп TPS не змінив темп")
	}
	raw, _ = os.ReadFile(settingsPath)
	if !strings.Contains(string(raw), `"tps"`) {
		t.Error("фліп TPS не зберігся у файл")
	}
	tps.next(nil)
	if gameTPS != gameTPSDefault {
		t.Errorf("другий фліп TPS не повернув дефолт: %d", gameTPS)
	}
	raw, _ = os.ReadFile(settingsPath)
	if strings.Contains(string(raw), `"tps"`) {
		t.Errorf("tps повернувся до дефолту, але ключ лишився: %s", raw)
	}
}

// TestSettingsInvalidTPSRejected — TPS має рівно два легальні значення; невідоме
// ВІДКИДАЄТЬСЯ (не клемпиться — 90 однаково далекий від обох), лишається дефолт.
func TestSettingsInvalidTPSRejected(t *testing.T) {
	sandboxSettings(t)

	gameTPS = gameTPSDefault
	if err := os.WriteFile(settingsPath, []byte(`{"tps": 90}`), 0644); err != nil {
		t.Fatal(err)
	}
	loadSettings()
	if gameTPS != gameTPSDefault {
		t.Errorf("нелегальний tps 90 застосувався: %d", gameTPS)
	}

	if err := os.WriteFile(settingsPath, []byte(`{"tps": 120}`), 0644); err != nil {
		t.Fatal(err)
	}
	loadSettings()
	if gameTPS != 120 {
		t.Errorf("легальний tps 120 не застосувався: %d", gameTPS)
	}
}
