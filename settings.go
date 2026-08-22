package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
)

// ==========================================================================
// [НАЛАШТУВАННЯ] Користувацькі налаштування у JSON — ШАРАМИ, як конфіг сервера.
//
//	1. дефолт у коді        (const antiAliasDefault, renderScaleDefault)
//	2. settings.json        (цей файл — ЛИШЕ ВІДХИЛЕННЯ від дефолтів)
//	3. рантайм-зміни сесії  (панель Tab) → save-back у шар 2
//
// Два принципи, обидва куплені шрамами цього проєкту:
//
// ВІДСУТНІСТЬ КЛЮЧА ≠ НУЛЬОВЕ ЗНАЧЕННЯ (PATCH, не PUT). Поля — вказівники:
// nil означає «не задано → дефолт коду». Користувач, що зберіг лише ss, не
// просив вимкнути AA.
//
// У ФАЙЛ ІДУТЬ ЛИШЕ ВІДХИЛЕННЯ. Значення == дефолту → ключ видаляється. Інакше
// сьогоднішні дефолти замерзли б у файлі, і майбутня зміна дефолту в коді тихо
// не долітала б — той самий клас підмін, що IDE-тека з вагами. Це та сама
// філософія, що HUD: показуємо (і зберігаємо) лише НЕтипове.
//
// Сюди потрапляють ЛИШЕ «живі» ручки комфорту (графіка, згодом звук/зум).
// Важелі експерименту (doubleDQN, frozenPolicy, sharedBrain…) — НІКОЛИ: їхні
// дефолти в коді стережуться тестами, і файл, що тихо відновлює «як було
// вчора», зруйнував би чесність замірів.
// ==========================================================================

// settingsPath — var, а не const: тести підставляють теку-пісочницю.
var settingsPath = "settings.json"

// userSettings — серіалізована форма. Вказівники = PATCH-семантика (див. шапку).
type userSettings struct {
	AA  *bool    `json:"aa,omitempty"`
	SS  *float32 `json:"ss,omitempty"`
	TPS *int     `json:"tps,omitempty"`
}

// loadSettings читає файл і накладає ВАЛІДНІ відхилення на рантайм-змінні.
// Кожна доля файлу — гучний рядок у консоль з АБСОЛЮТНИМ шляхом (урок ваг:
// відносний шлях означає залежність від робочої теки, і підміну має бути видно
// з першого рядка запуску).
func loadSettings() {
	abs, err := filepath.Abs(settingsPath)
	if err != nil {
		abs = settingsPath
	}
	raw, err := os.ReadFile(settingsPath)
	if err != nil {
		log.Printf("налаштування: %s — дефолти (файлу немає)", abs)
		return
	}
	var s userSettings
	if err := json.Unmarshal(raw, &s); err != nil {
		log.Printf("налаштування: %s — зіпсований JSON, дефолти (%v)", abs, err)
		return
	}
	if s.AA == nil && s.SS == nil && s.TPS == nil {
		log.Printf("налаштування: %s — відхилень немає, дефолти", abs)
		return
	}
	if s.AA != nil {
		antiAlias = *s.AA
		log.Printf("налаштування: %s — aa %v (типово %v)", abs, antiAlias, antiAliasDefault)
	}
	if s.SS != nil {
		// [ВАЛІДАЦІЯ] Файл — це ввід користувача, і в SS уже є шрам: ×12 клав гру
		// повз recover (гігабайтний буфер валить драйвер). У драбині лишаються лише
		// перевірені в грі щаблі — тож незнайоме значення КЛЕМПИТЬСЯ до найближчого,
		// а не застосовується.
		want := *s.SS
		got := nearestSSStep(want)
		renderScale = got
		if got != want {
			log.Printf("налаштування: %s — ss %g НЕ в драбині, взято найближчий щабель %g",
				abs, want, got)
		} else {
			log.Printf("налаштування: %s — ss %g (типово %g)", abs, renderScale, renderScaleDefault)
		}
	}
	if s.TPS != nil {
		// Темп має рівно два легальні значення — все в грі рахується в кадрах, і
		// довільний TPS означав би довільну швидкість симуляції. Невідоме значення
		// ВІДКИДАЄМО (не клемпимо: 90 однаково далекий від обох), лишаємо дефолт.
		if *s.TPS == 60 || *s.TPS == 120 {
			gameTPS = *s.TPS
			log.Printf("налаштування: %s — tps %d (типово %d)", abs, gameTPS, gameTPSDefault)
		} else {
			log.Printf("налаштування: %s — tps %d не підтримується (лише 60/120), дефолт %d",
				abs, *s.TPS, gameTPSDefault)
		}
	}
}

// saveSettings пише у файл ЛИШЕ відхилення від дефолтів; повна відповідність
// дефолтам дає порожній обʼєкт {} — файл-пустишку, що нічого не перекриває.
func saveSettings() {
	var s userSettings
	if antiAlias != antiAliasDefault {
		v := antiAlias
		s.AA = &v
	}
	if renderScale != renderScaleDefault {
		v := renderScale
		s.SS = &v
	}
	if gameTPS != gameTPSDefault {
		v := gameTPS
		s.TPS = &v
	}
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		log.Printf("налаштування: серіалізація: %v", err)
		return
	}
	if err := os.WriteFile(settingsPath, raw, 0644); err != nil {
		// Не ковтаємо: невдалий запис = мовчки втрачене налаштування (урок saveBrains).
		log.Printf("налаштування: запис %s: %v", settingsPath, err)
	}
}

// nearestSSStep — найближчий ПЕРЕВІРЕНИЙ щабель драбини суперсемплінгу.
func nearestSSStep(v float32) float32 {
	best := ssLadder[0]
	for _, s := range ssLadder[1:] {
		if abs32(s-v) < abs32(best-v) {
			best = s
		}
	}
	return best
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
