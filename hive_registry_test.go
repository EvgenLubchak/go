package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestExtinctTypeStillSaves — ваги типу, який ВИМЕР ПОВНІСТЮ, усе одно зберігаються.
//
// Пастка була тиха й дорога. Мережі були досяжні лише через g.units, а handleDeadUnits
// видаляє юніта, у якого скінчились повернення. Тип із скінченними поверненнями
// (у стражника Respawns: 1, тобто два життя) зникав зі зрізу разом зі своєю мережею —
// і на ESC його файл ваг не писався ВЗАГАЛІ.
//
// Жодного повідомлення при цьому не було: гра закривалась як завжди, просто сесія
// навчання того типу губилась. Найгірший вид вади — та, що не падає.
func TestExtinctTypeStillSaves(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "extinct.json")

	g := &Game{hive: map[string]*Net{}}
	n := NewNet()
	n.file = file
	g.hive[file] = n
	// Тіл цього типу немає ЗОВСІМ — саме той стан, у якому ваги й губились.
	g.units = nil

	g.saveBrains()
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("ваги вимерлого типу не збереглись: %v", err)
	}
}

// TestSaveWorksWithoutRegistry — режим sharedBrain=false зберігається так само.
//
// Друга половина фікса, без якої він був би регресом: коли кожен агент має ВЛАСНУ
// мережу, реєстр порожній за побудовою (netFor не кладе туди нічого), і збереження
// лише з реєстру не записало б нічого взагалі.
func TestSaveWorksWithoutRegistry(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "own.json")

	g := &Game{hive: map[string]*Net{}} // реєстр порожній
	n := NewNet()
	n.file = file
	g.units = []Pixel{{Brain: &Brain{net: n}}}

	g.saveBrains()
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("ваги агента з власною мережею не збереглись: %v", err)
	}
}

// TestRegistrySurvivesRestart — реєстр переживає рестарт, і мережа лишається ТА САМА.
//
// Це не лише про ваги: у Net живе БУФЕР ДОСВІДУ на 65536 переходів, який набирається
// тисячі кадрів. Підмінити мережу на рестарті означало б обнулити найдорожче.
func TestRegistrySurvivesRestart(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true

	cfg := ConfigWarden
	cfg.Count, cfg.WeightsFile = 2, filepath.Join(t.TempDir(), "w.json")
	unitRoster = []UnitConfig{cfg}

	g := &Game{difficulty: 1.0, player: newPlayer(), hive: map[string]*Net{}}
	g.units = newUnitsWithHive(g.hive)
	before := g.hive[cfg.WeightsFile]
	if before == nil {
		t.Fatal("реєстр не заповнився при створенні юнітів")
	}

	// Вимирання ДО рестарту: саме той випадок, коли збирання з тіл нічого не давало.
	g.units = nil
	g.restart()

	if after := g.hive[cfg.WeightsFile]; after != before {
		t.Error("після рестарту мережа інша — досвід і ваги втрачено")
	}
	if len(g.units) != cfg.Count {
		t.Errorf("рестарт створив %d юнітів замість %d", len(g.units), cfg.Count)
	}
	if g.units[0].Brain.net != before {
		t.Error("нові тіла отримали НЕ ту мережу, що в реєстрі")
	}
}
