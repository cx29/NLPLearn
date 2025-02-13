# 词性标注
## 中文 `jieba`
| 词性 | 含义 | 示例 |
|------|------|------|
| `n`  | 名词 | 人工智能、计算机、数据 |
| `nr` | 人名 | 李白、马云、乔布斯 |
| `ns` | 地名 | 北京、上海、纽约 |
| `nt` | 机构名 | 清华大学、微软公司 |
| `nz` | 其他专有名词 | 比特币、ChatGPT |
| `v`  | 动词 | 学习、处理、优化 |
| `vd` | 副动词 | 了解、思考 |
| `vn` | 名动词 | 开发、研究 |
| `a`  | 形容词 | 重要、智能、深度 |
| `ad` | 副形词 | 非常重要、极度智能 |
| `an` | 名形词 | 长度、高度 |
| `d`  | 副词 | 迅速、很、非常 |
| `m`  | 数词 | 一、十万、3.14 |
| `q`  | 量词 | 个、张、本 |
| `r`  | 代词 | 我、你、它、这、那个 |
| `p`  | 介词 | 在、从、对于 |
| `c`  | 连词 | 和、但、或者 |
| `u`  | 助词 | 的、了、着、过 |
| `ul` | 结构助词 | 的 |
| `uj` | 连接助词 | 的、地、得 |
| `y`  | 语气词 | 哈、吗、呢 |
| `e`  | 叹词 | 哎、哦、啊 |
| `o`  | 拟声词 | 哗啦、砰 |
| `h`  | 前缀 | 超、阿（如阿里） |
| `k`  | 后缀 | 界（如 IT 界） |
| `f`  | 方位词 | 上、下、左、右 |
| `t`  | 时间词 | 今天、明年、过去 |
| `i`  | 成语 | 一心一意、水滴石穿 |
| `l`  | 习惯用语 | 乱七八糟、画蛇添足 |
| `j`  | 简称略语 | 联大（联合国大会） |
| `x`  | 其它 | 网址、表情符号 |
| `w`  | 标点符号 | ，。！？、《》 |

## 英文 `nltk`

| **POS Tag** | **Description (描述)**                     | **Example (示例)**          |
|-------------|-------------------------------------------|----------------------------|
| **NN**      | 名词，单数或不可数名词                      | "dog", "water"             |
| **NNS**     | 名词，复数                                 | "dogs", "waters"           |
| **NNP**     | 专有名词，单数                             | "John", "New York"         |
| **NNPS**    | 专有名词，复数                             | "Americas", "Smiths"       |
| **PRP**     | 人称代词                                  | "I", "you", "he"           |
| **PRP$**    | 所有格代词                                | "my", "your", "his"        |
| **WP**      | 疑问代词                                  | "who", "what"              |
| **WP$**     | 所有格疑问代词                            | "whose"                    |
| **VB**      | 动词，原形                                | "eat", "run"               |
| **VBD**     | 动词，过去式                              | "ate", "ran"               |
| **VBG**     | 动词，现在分词或动名词                    | "eating", "running"        |
| **VBN**     | 动词，过去分词                            | "eaten", "run"             |
| **VBP**     | 动词，非第三人称单数现在时                | "eat", "run"               |
| **VBZ**     | 动词，第三人称单数现在时                  | "eats", "runs"             |
| **JJ**      | 形容词                                    | "beautiful", "tall"        |
| **JJR**     | 形容词，比较级                            | "better", "taller"         |
| **JJS**     | 形容词，最高级                            | "best", "tallest"          |
| **RB**      | 副词                                      | "quickly", "silently"      |
| **RBR**     | 副词，比较级                              | "better", "more quickly"   |
| **RBS**     | 副词，最高级                              | "best", "most quickly"     |
| **IN**      | 介词或从属连词                            | "in", "on", "because"      |
| **CC**      | 并列连词                                  | "and", "but", "or"         |
| **IN**      | 从属连词                                  | "although", "because"      |
| **DT**      | 限定词                                    | "the", "a", "some"         |
| **UH**      | 感叹词                                    | "wow", "ouch"              |
| **TO**      | 不定式标记（“to”）                        | "to go"                    |
| **EX**      | 存在句中的“there”                         | "There is a book."         |
| **FW**      | 外来词                                    | "bonjour"                  |
| **LS**      | 列表项标记                                | "1", "a"                   |
| **MD**      | 情态动词                                  | "can", "will", "must"      |
| **PDT**     | 前限定词                                  | "all", "half"              |
| **POS**     | 所有格结尾                                | "’s", "s’"                 |
| **RP**      | 小品词                                    | "up", "off"                |
| **SYM**     | 符号                                      | "$", "&"                   |
| **UH**      | 感叹词                                    | "hey", "oops"              |
| **WDT**     | 疑问限定词                                | "which", "what"            |