# QubePixel Block Configuration Guide

Руководство по конфигурированию блоков в JSON. Блоки определяются в `assets/blocks/` — каждый блок это отдельный `.json` файл.

## Структура блока JSON

```json
{
  "format_version": 2,
  "id": "blocks/my_block",
  "display_name": "My Block",
  "description": "Short description for UI",
  
  "color": [0.8, 0.8, 0.8],
  "material": {
    "albedo": [0.8, 0.8, 0.8],
    "roughness": 0.8,
    "metalness": 0.0,
    "ao": 1.0
  },
  
  "solid": true,
  "transparent": false,
  "biome_tint": false,
  
  "model": "path/to/model.json",
  "placement_mode": "look_horizontal",
  "default_rotation": 0,
  
  "emit_light": 0,
  "emission": {
    "emit_light": false,
    "light_color": [1.0, 1.0, 1.0],
    "light_strength": 0.0,
    "light_intensity": 0.0,
    "light_range": 24.0,
    "no_self_gi": false
  },
  
  "faces": {
    "all": { "color": [1.0, 1.0, 1.0], "texture": "textures/stone" },
    "top": { "color": [1.0, 1.0, 1.0], "texture": "textures/stone_top" },
    "sides": { "texture": "textures/stone_side" }
  }
}
```

## Основные поля

### Идентификация
- **`format_version`** (число): версия формата (текущая: 2)
- **`id`** (строка): уникальный идентификатор блока (используется в блок-реестре)
- **`display_name`** (строка): название для UI
- **`description`** (строка): описание блока

### Цвет и материалы
- **`color`** (RGB): базовый цвет вершин блока `[R, G, B]` (0.0–1.0)
- **`material`**: свойства PBR
  - `albedo`: диффузный цвет (обычно совпадает с основным)
  - `roughness`: шероховатость (0=глянец, 1=матовый)
  - `metalness`: металличность (0=неметалл, 1=металл)
  - `ao`: ambient occlusion множитель (0–1)
- **`biome_tint`** (bool): если true, цвет модулируется биомной тинтом (для травы, листвы)

### Физика и отрисовка
- **`solid`** (bool): является ли блок твёрдым (имеет коллизию)
- **`transparent`** (bool): прозрачен ли блок (сейчас не используется)

### 3D модель (опционально)
- **`model`** (строка): путь к Bedrock/Java модели блока (e.g., `models/furnace.json`)
  - Если не указано: блок отрисовывается как простой куб с текстурами из `faces`
  - Модель должна быть в формате Bedrock или Java Edition

### Поворот и ориентация
- **`placement_mode`** (строка): режим поворота при размещении блока
  - `"fixed"`: блок всегда в одном положении
  - `"look_horizontal"`: 4 ориентации на основе горизонтального направления взгляда
  - `"look_full"`: 6 ориентаций по полному направлению взгляда (вверх/вниз/4 стороны)
  - `"attach_horizontal"`: прилипает к одной из 4 боковых граней
  - `"attach_full"`: прилипает к любой из 6 граней куба
  - `"attach_full_rotatable"`: прилипает к любой грани + вращается на верх/нижних гранях
- **`default_rotation`** (число 0-13): значение поворота по умолчанию, если `placement_mode: "fixed"`

### Излучение света
- **`emit_light`** (число): DEPRECATED (используйте `emission`)
- **`emission`**: параметры самосвечения блока
  - `emit_light` (bool): излучает ли блок свет
  - `light_color`: цвет света `[R, G, B]`
  - `light_strength`: сила излучения (0–1)
  - `light_intensity`: интенсивность в пространстве (влияет на распространение через VCT)
  - `light_range`: максимальная дальность распространения света (блоки)
  - `no_self_gi`: если true, блок не получает GI от собственного излучения

### Текстуры и цвета граней
- **`faces`** (объект): определение текстур/цветов отдельных граней
  ```json
  "faces": {
    "all": { "color": [1.0, 1.0, 1.0], "texture": "stone" },
    "top": { "texture": "stone_top" },
    "bottom": { "color": [0.5, 0.5, 0.5] },
    "sides": { "texture": "stone_side" },
    "north": { "texture": "stone_front" },
    "south": { "texture": "stone_back" },
    "east": { "texture": "stone_east" },
    "west": { "texture": "stone_west" }
  }
  ```

  **Приоритет поиска текстуры** (порядок fallback):
  - Для верхней/нижней граней: `top`/`bottom` → `all` → базовый цвет
  - Для боковых граней (N/S/E/W): конкретная сторона → `sides` → `all` → базовый

---

## Режимы поворота (Placement Modes)

### `fixed` (фиксированный)
Блок всегда в одной ориентации. `default_rotation` определяет, какое значение из таблицы ROT_FACE_MAP использовать (обычно 0 или 2 для направления south).

**Примеры**: каменная плита, глина, травяной блок

```json
"placement_mode": "fixed",
"default_rotation": 0
```

### `look_horizontal` (горизонтальный вид) ✅ РЕАЛИЗОВАНО
Блок ориентируется на основе горизонтального направления взгляда игрока (N/S/E/W). 4 ориентации:
- **Rotation 1**: North (-Z) — блок "смотрит" на север
- **Rotation 2**: South (+Z) — блок "смотрит" на юг (default)
- **Rotation 3**: East (+X) — блок "смотрит" на восток
- **Rotation 4**: West (-X) — блок "смотрит" на запад

**Для кубов**: меняет, какая текстура отображается на каком грани (через ROT_FACE_MAP).
**Для моделей Bedrock**: применяет Y-ось ротацию модели (+90° для east, -90° для west, 180° для north).

**Примеры**: печь, верстак, сундук (все имеют "перёд" и "сзади"), редстоун-печь

```json
"placement_mode": "look_horizontal",
"default_rotation": 0
```

⚠️ **Примечание про брёвна**: `oak_log` имеет одинаковую текстуру на всех 4 боковых гранях, поэтому визуально кажется, что ротация не работает. Чтобы брёвна лежали на боку (с колец на боковых гранях), нужен режим `look_axis` (не реализован).

### `look_full` (полный вид) ✅ РЕАЛИЗОВАНО (базовая версия)
Блок ориентируется по полному направлению взгляда, включая вверх/вниз. 6 основных ориентаций:
- **Rotation 0**: Default (floor)
- **Rotation 1-4**: N/S/E/W (как в `look_horizontal`)
- **Rotation 5**: Upside-down (потолок)

**Для моделей**: поддерживается только Y ротация (вверх/вниз не применяют дополнительный pitch).

**Примеры**: блоки, которые могут быть установлены на потолок

```json
"placement_mode": "look_full",
"default_rotation": 0
```

### `attach_horizontal` ✅ РЕАЛИЗОВАНО
Блок прилипает к одной из 4 боковых граней другого блока. На верх/низ щелчок использует горизонтальное направление взгляда.

- **North wall**: rotation 6
- **South wall**: rotation 7
- **East wall**: rotation 8
- **West wall**: rotation 9

**Примеры**: рычаги, кнопки, факелы (attach_horizontal предусмотрен, но используют более специальные режимы)

```json
"placement_mode": "attach_horizontal",
"default_rotation": 0
```

### `attach_full` ✅ РЕАЛИЗОВАНО
Блок может прилипать к любой из 6 граней куба (верх, низ, 4 стороны). Нет дополнительного вращения на верх/низ.

- **Top (floor)**: rotation 0
- **Bottom (ceiling)**: rotation 5
- **North/South/East/West walls**: rotation 6-9

**Примеры**: редстоун-проводка (может быть на стенах)

```json
"placement_mode": "attach_full",
"default_rotation": 0
```

### `attach_full_rotatable` ✅ РЕАЛИЗОВАНО
Блок прилипает к любой из 6 граней И может вращаться:
- На верхней/нижней грани: выбирает одну из 4 ориентаций по горизонтальному взгляду (rotations 0, 2, 10, 11)
- На стенах: фиксированная ориентация (6-9)

**Примеры**: лестницы, плиты (когда помещены на потолок, ориентируются по взгляду)

```json
"placement_mode": "attach_full_rotatable",
"default_rotation": 0
```

---

## Таблица ротаций (ROT_FACE_MAP)

Каждая ротация переопределяет, какая локальная грань блока отображается на каждой мировой грани.

**Индексы граней**:
- 0 = +X (East/восток)
- 1 = -X (West/запад)
- 2 = +Y (Top/верх)
- 3 = -Y (Bottom/низ)
- 4 = +Z (South/юг)
- 5 = -Z (North/север)

**Текущая таблица** (в `block.rs`):
```
 0: [0, 1, 2, 3, 4, 5]   // default (floor, south-facing)
 1: [1, 0, 2, 3, 5, 4]   // north-facing (180° Y rotation)
 2: [0, 1, 2, 3, 4, 5]   // south-facing (= default)
 3: [4, 5, 2, 3, 1, 0]   // east-facing (+90° Y rotation)
 4: [5, 4, 2, 3, 0, 1]   // west-facing (-90° Y rotation)
 5: [0, 1, 3, 2, 5, 4]   // upside-down (180° X rotation)
 6-13: ...              // стенные и потолочные ориентации
```

---

## Примеры конфигов

### Простой куб (камень)
```json
{
  "format_version": 2,
  "id": "stone",
  "display_name": "Stone",
  "color": [0.6, 0.6, 0.6],
  "material": { "albedo": [0.6, 0.6, 0.6], "roughness": 0.9, "metalness": 0.0, "ao": 1.0 },
  "solid": true,
  "placement_mode": "fixed",
  "faces": { "all": { "texture": "stone" } }
}
```

### Ориентируемый блок (печь)
```json
{
  "format_version": 2,
  "id": "furnace",
  "display_name": "Furnace",
  "color": [0.3, 0.3, 0.3],
  "solid": true,
  "placement_mode": "look_horizontal",
  "faces": {
    "top": { "texture": "furnace_top" },
    "bottom": { "texture": "furnace_bottom" },
    "sides": { "texture": "furnace_side" },
    "south": { "texture": "furnace_front" }
  }
}
```

### Модель Bedrock с поворотом
```json
{
  "format_version": 2,
  "id": "redstone_furnace",
  "display_name": "Redstone Furnace",
  "model": "models/redstone_furnace.json",
  "solid": true,
  "placement_mode": "look_horizontal",
  "default_rotation": 0
}
```

### Светящийся блок (глоустоун)
```json
{
  "format_version": 2,
  "id": "glowstone",
  "display_name": "Glowstone",
  "color": [1.0, 1.0, 0.8],
  "solid": true,
  "placement_mode": "fixed",
  "emission": {
    "emit_light": true,
    "light_color": [1.0, 1.0, 0.7],
    "light_strength": 0.8,
    "light_intensity": 15.0,
    "light_range": 32.0
  },
  "faces": { "all": { "texture": "glowstone" } }
}
```

### Брёвна (с биом-тинтом, но без боковой укладки)
```json
{
  "format_version": 2,
  "id": "oak_log",
  "display_name": "Oak Log",
  "color": [0.45, 0.30, 0.15],
  "material": { "albedo": [0.45, 0.30, 0.15], "roughness": 0.95, "metalness": 0.0, "ao": 1.0 },
  "solid": true,
  "placement_mode": "look_horizontal",
  "faces": {
    "top": { "color": [0.55, 0.40, 0.22], "texture": "oak_log_top" },
    "bottom": { "color": [0.55, 0.40, 0.22], "texture": "oak_log_top" },
    "sides": { "color": [0.42, 0.28, 0.13], "texture": "oak_log_side" }
  }
}
```

---

## Режимы, которые не реализованы

### `look_axis` (будущее)
**Описание**: Блок выравнивается вдоль оси в направлении взгляда (X, Y, Z).

**Зачем**: Для брёвен, которые должны иметь кольца на торцах в направлении укладки. В Minecraft брёвна это работают через свойство `axis: [x|y|z]`.

**Как будет работать**:
- Взгляд горизонтально (N/S/E/W): брёвно лежит горизонтально (торцы на боков, кора на верху/низу)
- Взгляд вверх/вниз: брёвно вертикально (кора на всех боков, кольца на верху/низу)

**Требуемые изменения**:
- Новая `PlacementMode` в `block.rs`
- Функция вычисления оси в `player.rs`
- ROT_FACE_MAP значения, которые свопят боковые грани с верхом/низом

---

## Возможные будущие режимы

### `look_axis_free` (полная 3D осевая укладка)
Брёвна укладываются точно в направлении взгляда (вперёд/назад, влево/вправо, вверх/вниз) с максимальной точностью.

### `slot_attachable` (врезные блоки)
Блок занимает одну из "щелей" на другом блоке (например, как рельсы могут быть на краю платформы).

### `rotation_with_pitch` (полная 3D ротация)
Блок может иметь произвольную pitch ротацию (наклоны), не только yaw. Для моделей — поддержка X/Z ротаций в дополнение к Y.

### `biome_specific` (биом-зависимый внешний вид)
Внешний вид блока кардинально отличается в зависимости от биома (например, песок в пустыне vs снег в тундре).

---

## Отладка

### Проверка конфигурации
1. JSON валидность: используйте онлайн JSON validator
2. Путь до текстур: проверьте `assets/textures/` (или используйте встроенные текстуры вроде `white`, `checker`)
3. ID блока: должен совпадать с входом в `block_registry.json`

### Логирование
При загрузке блока в логе появится:
```
[BlockRegistry] Loaded block: my_block (id=X)
```

Если есть проблема с текстурой:
```
[TextureAtlas] Texture not found: my_texture
```

### Тестирование в игре
1. Поместите JSON в `assets/blocks/`
2. Добавьте ID в `assets/blocks/block_registry.json`
3. Перезапустите игру
4. Создайте инвентарь через дебаг и выберите блок из списка

---

## Часто задаваемые вопросы

**Q: Почему мой блок не поворачивается?**
A: Проверьте:
1. `placement_mode` установлен на что-то кроме `fixed`
2. У блока есть отличающиеся текстуры/цвета на разных гранях (иначе ротация невидима)
3. Для моделей Bedrock: убедитесь, что модель загружена (проверьте логи)

**Q: Как сделать, чтобы текстура была на всех гранях одинаковой?**
A: Используйте `"all": { "texture": "my_texture" }` в `faces`.

**Q: Могу ли я иметь разные цвета и текстуры на одной грани?**
A: Нет, грань имеет одну текстуру и один цвет. Для большей сложности используйте Bedrock модель.

**Q: Как брёвна могут лежать на боку?**
A: Нужен режим `look_axis` (не реализован). Сейчас брёвна с `look_horizontal` всегда имеют кольца на верху/низу.

**Q: Где найти форматы моделей Bedrock/Java?**
A: 
- Bedrock: используйте объекты `geometry` из `.json` моделей
- Java: используйте объекты `elements` из `.json` моделей
- QubePixel поддерживает оба через `BlockModelParser`
