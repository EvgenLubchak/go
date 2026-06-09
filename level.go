package main

// Тайлова мапа рівня.
// Розмір: 30×17 тайлів (screenWidth=1680, screenHeight=960, pixelSize=55).
//
// Символи:
//   '#' = стіна
//   ' ' = підлога
//   'P' = точка старту гравця (один символ)
//   'E' = точка спавну ворога (скільки завгодно)
//
// P і E трактуються як підлога при колізіях.
// Кожен рядок — рівно boidMapW (30) символів, рядків — boidMapH (17).

// [GO: SLICE OF STRINGS AS 2D MAP]
// Зручно редагувати: ти бачиш рівень в редакторі як справжню сітку.
// Для колізій парсимо у 2D bool-масив один раз при старті (init).
var levelLayout = []string{
	//         1111111111222222222
	//1234567890123456789012345678901
	"                              ",
	"#####             ##          ",
	"    #             #  E        ",
	"  P #            ##           ",
	"    #                         ",
	" ####                         ",
	" #                            ",
	" #                            ",
	" #            E               ",
	" #                            ",
	"##                            ",
	"      #####  #   #  #   #     ",
	"        #    #   #  #  #      ",
	"        #    # E #  ###       ",
	"        #    #   #  #  #      ",
	"        #    ###### #   #     ",
	"          E       #   E       ",
}

// SpawnPoint — піксельна координата точки спавну.
type SpawnPoint struct{ X, Y float32 }

var (
	tileMap     [boidMapH][boidMapW]bool
	playerSpawn = SpawnPoint{X: screenWidth/2 - pixelSize/2, Y: screenHeight/2 - pixelSize/2} // дефолт — центр
	enemySpawns []SpawnPoint
)

func init() {
	for row, line := range levelLayout {
		if row >= boidMapH {
			break
		}
		for col, ch := range line {
			if col >= boidMapW {
				break
			}
			switch ch {
			case '#':
				tileMap[row][col] = true
			case 'P':
				// [GO: STRUCT LITERAL] — нова структура з полями X і Y
				playerSpawn = SpawnPoint{
					X: float32(col * pixelSize),
					Y: float32(row * pixelSize),
				}
			case 'E':
				// [GO: APPEND] — додаємо в динамічний slice
				enemySpawns = append(enemySpawns, SpawnPoint{
					X: float32(col * pixelSize),
					Y: float32(row * pixelSize),
				})
			}
		}
	}
}

// isWallAt перевіряє чи тайл [col, row] є стіною.
func isWallAt(col, row int) bool {
	if col < 0 || col >= boidMapW || row < 0 || row >= boidMapH {
		return true // за межами екрану = стіна
	}
	return tileMap[row][col]
}

// isWallRect перевіряє 4 кути пікселя на стіну.
// Out-of-bounds = стіна → вороги відбиваються від країв поля.
func isWallRect(x, y float32) bool {
	corners := [4][2]float32{
		{x, y}, {x + pixelSize - 1, y},
		{x, y + pixelSize - 1}, {x + pixelSize - 1, y + pixelSize - 1},
	}
	for _, c := range corners {
		if isWallAt(int(c[0])/pixelSize, int(c[1])/pixelSize) {
			return true
		}
	}
	return false
}

// isInteriorWallRect — тільки внутрішні тайли, ігнорує межі екрану.
// Для гравця: дозволяє wrap-around через краї, але блокує стіни всередині мапи.
func isInteriorWallRect(x, y float32) bool {
	corners := [4][2]float32{
		{x, y}, {x + pixelSize - 1, y},
		{x, y + pixelSize - 1}, {x + pixelSize - 1, y + pixelSize - 1},
	}
	for _, c := range corners {
		col := int(c[0]) / pixelSize
		row := int(c[1]) / pixelSize
		if col < 0 || col >= boidMapW || row < 0 || row >= boidMapH {
			continue // за межами = не стіна для гравця
		}
		if tileMap[row][col] {
			return true
		}
	}
	return false
}
