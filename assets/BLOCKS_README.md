# Структура системы блоков QubePixel

## Обзор
Система загрузки блоков теперь использует файловую структуру, аналогичную Minecraft. 
Для добавления нового блока нужно выполнить два шага.

## Структура файлов
```
assets/
├── block_registry.json      # Реестр всех блоков (список ID)
└── blocks/                  # Папка с определениями блоков
    ├── grass.json
    ├── dirt.json
    └── stone.json
```

## Шаг 1: Добавить блок в реестр
Откройте `assets/block_registry.json` и добавьте ID нового блока в массив `blocks`:

```json
{
  "blocks": [
    "grass",
    "dirt",
    "stone",
    "sand"  // ← Добавьте новую строку (не забудьте запятую после предыдущего элемента)
  ]
}
```

## Шаг 2: Создать файл определения блока
Создайте файл `assets/blocks/sand.json` со следующей структурой:

```json
{
  "id": "sand",                    // Обязательное поле: должно совпадать с именем файла
  "name": "Sand Block",            // Необязательно: отображаемое имя
  "description": "A block of sand", // Необязательно: описание
  "color": [0.76, 0.70, 0.50],     // Обязательное поле: RGB цвет [0.0-1.0]
  "solid": true,                   // Обязательное поле: твёрдый ли блок
  "transparent": false,            // Обязательное поле: прозрачный ли блок
  "hardness": 0.5,                 // Необязательно: твёрдость (по умолчанию 0.5)
  "blast_resistance": 0.5,         // Необязательно: взрывостойкость (по умолчанию 0.5)
  "light_emission": 0.0,           // Необязательно: излучение света 0.0-1.0 (по умолчанию 0.0)
  "friction": 0.6,                 // Необязательно: трение (по умолчанию 0.6)
  "slipperiness": 0.6,             // Необязательно: скольжение (по умолчанию 0.6)
  "requires_tool": false,          // Необязательно: требует ли инструмент (по умолчанию false)
  "stack_size": 64,                // Необязательно: размер стака (по умолчанию 64)
  "textures": {                    // Необязательно: текстуры блока
    "all": "blocks/sand.png"       // Можно указать одну текстуру для всех сторон
  }
}
```

## Поля блока

### Обязательные поля
| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | Уникальный идентификатор блока. Должен совпадать с именем файла |
| `color` | [f32; 3] | RGB цвет в формате [R, G, B] где значения от 0.0 до 1.0 |
| `solid` | bool | Является ли блок твёрдым (препятствует движению) |
| `transparent` | bool | Является ли блок прозрачным (влияет на освещение) |

### Необязательные поля (имеют значения по умолчанию)
| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `name` | string | "" | Человекочитаемое название блока |
| `description` | string | "" | Описание блока |
| `hardness` | f32 | 0.5 | Время разрушения блока |
| `blast_resistance` | f32 | 0.5 | Сопротивление взрывам |
| `light_emission` | f32 | 0.0 | Уровень излучаемого света (0.0-1.0) |
| `friction` | f32 | 0.6 | Коэффициент трения |
| `slipperiness` | f32 | 0.6 | Коэффициент скольжения |
| `requires_tool` | bool | false | Требуется ли инструмент для добычи |
| `stack_size` | u8 | 64 | Максимальный размер стака (1-99) |
| `textures` | object | null | Текстуры блока |

## Текстуры

Объект `textures` поддерживает следующие поля:

| Поле | Описание |
|------|----------|
| `all` | Единая текстура для всех сторон (используется если другие не указаны) |
| `top` | Текстура верхней грани |
| `bottom` | Текстура нижней грани |
| `side` | Текстура боковых граней |
| `front` | Текстура передней грани |
| `back` | Текстура задней грани |

### Примеры конфигурации текстур

**Одна текстура для всех сторон:**
```json
"textures": {
  "all": "blocks/stone.png"
}
```

**Разные текстуры для верха/низа/боков:**
```json
"textures": {
  "top": "blocks/grass_block_top.png",
  "bottom": "blocks/dirt.png",
  "side": "blocks/grass_block_side.png"
}
```

**Полная конфигурация (все 6 сторон):**
```json
"textures": {
  "top": "blocks/furnace_top.png",
  "bottom": "blocks/furnace_bottom.png",
  "front": "blocks/furnace_front.png",
  "back": "blocks/furnace_back.png",
  "side": "blocks/furnace_side.png"
}
```

## Примеры готовых блоков

### Песок (sand.json)
```json
{
  "id": "sand",
  "name": "Sand",
  "color": [0.76, 0.70, 0.50],
  "solid": true,
  "transparent": false,
  "hardness": 0.5,
  "textures": {
    "all": "blocks/sand.png"
  }
}
```

### Дерево (wood.json)
```json
{
  "id": "wood",
  "name": "Oak Wood",
  "description": "A block of oak wood",
  "color": [0.40, 0.27, 0.15],
  "solid": true,
  "transparent": false,
  "hardness": 2.0,
  "blast_resistance": 2.0,
  "requires_tool": false,
  "textures": {
    "top": "blocks/log_oak_top.png",
    "side": "blocks/log_oak_side.png"
  }
}
```

### Светящийся камень (glowstone.json)
```json
{
  "id": "glowstone",
  "name": "Glowstone",
  "color": [0.80, 0.70, 0.40],
  "solid": true,
  "transparent": false,
  "hardness": 0.3,
  "light_emission": 1.0,
  "textures": {
    "all": "blocks/glowstone.png"
  }
}
```

### Лёд (ice.json)
```json
{
  "id": "ice",
  "name": "Ice",
  "color": [0.65, 0.80, 0.90],
  "solid": true,
  "transparent": true,
  "hardness": 0.5,
  "slipperiness": 0.98,
  "textures": {
    "all": "blocks/ice.png"
  }
}
```

## Отладка

При запуске игры в логах вы увидите сообщения о загрузке блоков:
```
[BlockRegistry][load] Loading registry from "assets/block_registry.json"
[BlockRegistry][load] Loading block 'grass' from "assets/blocks/grass.json"
[BlockRegistry][load] Registered block 'grass' => id 1 (solid=true, transparent=false)
[BlockRegistry][load] Total block types loaded: 3
```

## Возможные ошибки

1. **Файл block_registry.json не найден**
   - Убедитесь, что файл существует в папке `assets/`
   - Проверьте права доступа к файлу

2. **Файл блока не найден**
   - Убедитесь, что файл существует в `assets/blocks/<id>.json`
   - Проверьте соответствие ID в реестре и имени файла

3. **Несоответствие ID**
   - Поле `id` внутри JSON файла должно совпадать с именем файла
   - Например, в файле `sand.json` должно быть `"id": "sand"`

4. **Неверный формат JSON**
   - Проверьте синтаксис JSON (запятые, кавычки, скобки)
   - Используйте JSON валидатор

5. **Отсутствуют обязательные поля**
   - Проверьте наличие полей: `id`, `color`, `solid`, `transparent`
