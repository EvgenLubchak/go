package main

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// randTransition — випадковий перехід для наповнення буфера в тестах циклу сесії.
func randTransition() transition {
	var t transition
	for i := 0; i < brainInputs; i++ {
		t.s[i] = rand.Float32()*2 - 1
		t.s2[i] = rand.Float32()*2 - 1
	}
	t.a = rand.Intn(brainActions)
	t.r = rand.Float32()*2 - 1
	return t
}

// TestEscCycleRoundTripsWardenWeights — ПОВНИЙ протокол «ESC → перезапуск» для
// стражника: ваги, які гра зберегла на виході, наступна гра завантажує БІТ-У-БІТ,
// і політика (Q на будь-якому стані) ідентична до перезаходу.
//
// Написано на живу підозру: «після перезаходу через ESC стражник починає спамити
// ухилення». Якщо цей тест зелений — самі ВАГИ тут ні до чого, і причину треба
// шукати в тому, що НЕ зберігається (буфер досвіду, свіжість target-мережі,
// памʼять агентів) — див. TestRestartKeepsBufferEscCycleDoesNot нижче.
func TestEscCycleRoundTripsWardenWeights(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true

	cfg := ConfigWarden
	cfg.Count = 2
	cfg.WeightsFile = filepath.Join(t.TempDir(), "warden.json")
	unitRoster = []UnitConfig{cfg}

	// ГРА А: свіжа мережа, трохи тренування — щоб ваги відійшли від ініціалізації
	// (інакше порівняння нижче було б порожнім: збіг «нічого з нічим»).
	gA := &Game{difficulty: 1.0, player: newPlayer(), hive: map[string]*Net{}}
	gA.units = newUnitsWithHive(gA.hive)
	netA := gA.units[0].Brain.net
	if netA.replayCap != cfg.ReplayCap {
		t.Fatalf("свіжій мережі не застосувався ReplayCap типу: %d замість %d",
			netA.replayCap, cfg.ReplayCap)
	}
	initW1 := netA.W1
	for i := 0; i < 300; i++ {
		netA.remember(randTransition())
	}
	for k := 0; k < 50; k++ {
		netA.train(qBatch)
	}
	if netA.W1 == initW1 {
		t.Fatal("тренування не зрушило ваги — порівняння нижче було б порожнім")
	}

	gA.saveBrains() // те, що робить ESC

	// ГРА Б: чистий процес — реєстр порожній, мережа мусить прийти з файлу.
	hiveB := map[string]*Net{}
	unitsB := newUnitsWithHive(hiveB)
	netB := unitsB[0].Brain.net
	if netB == netA {
		t.Fatal("гра Б отримала ту саму мережу в памʼяті — цикл не відтворює перезапуск")
	}

	// 1) Усі стек-ваги біт-у-біт (масиви в Go порівнюються цілком).
	if netB.W1 != netA.W1 || netB.B1 != netA.B1 ||
		netB.W2 != netA.W2 || netB.B2 != netA.B2 ||
		netB.W3 != netA.W3 || netB.B3 != netA.B3 {
		t.Fatal("ваги після циклу ESC→завантаження не збігаються біт-у-біт")
	}

	// 2) Горизонт, стеля цінності й контракт памʼяті — ті самі.
	if netB.gamma != netA.gamma || netB.clip != netA.clip || netB.mem != netA.mem {
		t.Fatalf("контракт/горизонт розійшовся: γ %v→%v, clip %v→%v, mem %+v→%+v",
			netA.gamma, netB.gamma, netA.clip, netB.clip, netA.mem, netB.mem)
	}
	if netB.file != cfg.WeightsFile {
		t.Fatalf("мережа не памʼятає свій файл: %q", netB.file)
	}
	if netB.replayCap != cfg.ReplayCap {
		t.Fatalf("завантаженій мережі не застосувався ReplayCap типу: %d", netB.replayCap)
	}

	// 3) Поведінка ідентична: Q збігаються точно на випадкових станах, і target
	//    після завантаження дорівнює живим вагам (документована поведінка).
	for k := 0; k < 50; k++ {
		var s [brainInputs]float32
		for i := range s {
			s[i] = rand.Float32()*2 - 1
		}
		qa, _, _ := netA.forwardQ(s)
		qb, _, _ := netB.forwardQ(s)
		if qa != qb {
			t.Fatalf("Q розійшлись після циклу: %v проти %v", qa, qb)
		}
		if qt := netB.forwardQTarget(s); qt != qb {
			t.Fatalf("target після завантаження не синхронізовано з живими вагами")
		}
	}

	// 4) А ось буфер досвіду НЕ переживає цикл — свідомо (він не зберігається).
	//    Це не вада тесту, це ГОЛОВНА різниця між R і ESC: див. тест нижче.
	if netB.replayLen() != 0 {
		t.Fatalf("буфер після завантаження мав би бути порожнім, а має %d", netB.replayLen())
	}
}

// TestRestartKeepsBufferEscCycleDoesNot — закріплює АСИМЕТРІЮ, яку видно в грі:
//
//	R (рестарт)         → та сама мережа, буфер досвіду ЖИВИЙ
//	ESC → перезапуск    → ваги ті самі, але буфер ПОРОЖНІЙ
//
// Чому це важливо: одразу після перезаходу навчання йде по крихітному й максимально
// корельованому буферу (qMinReplay = 200 при 18 стражниках набирається за ~12 кадрів,
// далі qBatch оновлень щокадру семплюють ті самі свіжі переходи десятки разів). Якщо
// в ці секунди стражників бʼють — política різко хилиться в бік «болю зараз».
// Після R такого шквалу немає: кільце повне старої різноманітної історії.
func TestRestartKeepsBufferEscCycleDoesNot(t *testing.T) {
	savedRoster, savedShared := unitRoster, sharedBrain
	defer func() { unitRoster, sharedBrain = savedRoster, savedShared }()
	sharedBrain = true

	cfg := ConfigWarden
	cfg.Count = 2
	cfg.WeightsFile = filepath.Join(t.TempDir(), "warden.json")
	unitRoster = []UnitConfig{cfg}

	g := &Game{difficulty: 1.0, player: newPlayer(), hive: map[string]*Net{}}
	g.units = newUnitsWithHive(g.hive)
	net := g.units[0].Brain.net
	for i := 0; i < 300; i++ {
		net.remember(randTransition())
	}

	g.restart()
	if got := g.units[0].Brain.net; got != net {
		t.Fatal("рестарт підмінив мережу — реєстр не спрацював")
	}
	if n := net.replayLen(); n != 300 {
		t.Fatalf("рестарт втратив буфер досвіду: %d із 300", n)
	}

	g.saveBrains()
	fresh := newUnitsWithHive(map[string]*Net{})
	if n := fresh[0].Brain.net.replayLen(); n != 0 {
		t.Fatalf("несподівано: буфер пережив цикл ESC→завантаження (%d)", n)
	}
}
