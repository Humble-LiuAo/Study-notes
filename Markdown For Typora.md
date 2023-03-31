# Markdown For Typora

## Overview

**Markdown** is created by [Daring Fireball](http://daringfireball.net/); the original guideline is [here](http://daringfireball.net/projects/markdown/syntax). Its syntax, however, varies between different parsers or editors. **Typora** is using [GitHub Flavored Markdown][GFM].

[toc]

## Block Elements

### Paragraph and line breaks

```markdown
1.按一次"return"即可创建新段落
2.按"shift"+"return"以创建单行中断。大多数其他减值解析器将忽略单行中断，因此，为了让其他减值解析器识别您的线路中断，您可以在行的末尾留出两个空格，或插入"<br/>"。
```

### Headers

```markdown
标题在行的开头使用 1-6 个哈希字符（'#'），对应于标题级别 1-6。
例如：
# This is an H1

## This is an H2

###### This is an H6
```

### Blockquotes

```markdown
Markdown 使用电子邮件式>字符进行阻止引用。它们被呈现为:
> This is a blockquote with two paragraphs. This is first paragraph.
>
> This is second pragraph. Vestibulum enim wisi, viverra nec, fringilla in, laoreet vitae, risus.



> This is another blockquote with one paragraph. There is three empty line to seperate two blockquote.

在 Typora 中，输入">"，然后输入您的报价内容将生成报价块。Typora 将为您插入适当的">"或线路中断。嵌套块报价（另一个区块报价中的块报价），通过添加额外的">"级别。
```

### Lists

```markdown
输入"* 列表项目 " 将创建一个未排序的列表 -"*"符号可以替换为"+"或"-"。   输入 '1.列表项目 ' 将创建一个订单列表 - 其标记源代码如下：
## un-ordered list
*   Red
*   Green
*   Blue

## ordered list
1.  Red
2. 	Green
3.	Blue
```

### Task List

```markdown
任务列表是标有 [] 或 [x] （不完整或已完成） 的项目列表。例如：
- [ ] a task list item
- [ ] list syntax required
- [ ] normal **formatting**, @mentions, #1234 refs
- [ ] incomplete
- [x] completed
```

### (Fenced) Code Blocks

````markdown
Typora 只支持 fences in GitHub Flavored Markdown。不支持 Markdown 中的原始代码块。   使用 fences 很容易：输入```和按 'return'。在```之后添加可选语言标识符，我们将通过语法加亮来运行它：
Here's an example:

```js
function test() {
  console.log("notice the blank line before this function?");
}
```

syntax highlighting:
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```
````

### Math Blocks

```markdown
您可以使用**MathJax**来渲染[LaTeX]数学表达式。
要添加数学表达，输入"$$"并按"return"键。这将触发一个接受 *Tex/LaTex* 源的输入字段。例如：
```

$$
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
\end{vmatrix}
$$

```markdown
In the markdown source file, the math block is a *LaTeX* expression wrapped by a pair of ‘$$’ marks:
$$
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
\end{vmatrix}
$$
```

You can find more details [here](https://support.typora.io/Math/).

### Tables

输入`| First Header  | Second Header |` "并按`"return"`键。这将创建一个带有两列的表。   创建表后，将焦点放在该表上将打开表的工具栏，您可以调整表大小、对齐或删除表。您还可以使用上下文菜单复制和添加/删除单个列/行。   表的完整语法如下所述，但无需详细了解完整的语法，因为表的标记源代码由 Typora 自动生成。   

在 markdown 源代码中，它们看起来像：

```markdown
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
```

您还可以在表中包含 inline Markdown 标记，如链接、粗体、斜体或删除线（links, bold, italics, or strikethrough）。   

最后，通过在头行中包含冒号  (`:`)，您可以将该列中的文本定义为左对齐、右对齐或中对齐：

```markdown
| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ |:---------------:| -----:|
| col 3 is      | some wordy text | $1600 |
| col 2 is      | centered        |   $12 |
| zebra stripes | are neat        |    $1 |
```

最左边的冒号表示左对齐的列；最右边的冒号表示列是右对齐的；两边的冒号表示是中心对齐的列。

### Footnotes

```markdown
You can create footnotes like this[^footnote].

[^footnote]: Here is the *text* of the **footnote**.
```

will produce:

You can create footnotes like this[^footnote].

[^footnote]: Here is the *text* of the **footnote**.

将鼠标悬停在“脚注”上标上，可以查看脚注的内容。

### Horizontal Rules

在空行输入 ` *** ` 或 `---`，并按 `return` 将画出一条水平线。

### YAML Front Matter

Typora 现在支持[YAML Front Matter](http://jekyllrb.com/docs/frontmatter/)。在文章的顶部输入`---`，然后按 `Return` 引入元数据块。或者，您可以从 Typora 的顶部菜单中插入元数据块。

### Table of Contents(TOC)

输入`[toc]`并按`Return`键。这将创建一个“目录”部分。TOC从文档中提取所有标题，当您添加到文档时，它的内容会自动更新。

## Span Elements

Span元素将在输入后立即被解析和呈现。将光标移动到这些span元素的中间会将这些元素展开为markdown源。下面是每个span元素的语法解释

### Links

Markdown支持两种样式的链接:内联和引用。

在这两种样式中，链接文本由 [方括号] 分隔。

要创建内联链接，请在链接文本的右方括号后立即使用一组规则括号。在圆括号内，将URL放置在您希望链接指向的位置，并为链接提供一个可选的标题，用引号括起来。例如:

```markdown
This is [an example](http://example.com/ "Title") inline link.

[This link](http://example.net/) has no title attribute.
```

will produce:

This is [an example](http://example.com/ "Title") inline link. (`<p>This is <a href="http://example.com/" title="Title">`)

[This link](http://example.net/) has no title attribute.. (`<p><a href="http://example.net/">This link</a> has no`)

#### Internal Links

**您可以将 href 设置为headers**，这将创建一个书签，允许您单击后跳转到该部分。

例如: 命令(在Windows上:Ctrl) +单击此 [This link](#block-elements) 将跳转到标题块元素。要查看如何编写，请移动光标或按下⌘键单击该链接以将元素展开为 markdown 源。

#### Reference Links

参考样式的链接使用第二组方括号，在里面你可以选择一个标签来标识链接:

```markdown
This is [an example][id] reference-style link.

Then, anywhere in the document, you define your link label on a line by itself like this:

[id]: http://example.com/  "Optional Title Here"
```

will produce

This is [an example][id] reference-style link.

Then, anywhere in the document, you define your link label on a line by itself like this:

[id]: http://example.com/  "Optional Title Here"

隐式链接名称快捷方式允许您省略链接的名称，在这种情况下链接文本本身将用作名称。只需使用一组空的方括号-例如，链接单词“Google”到 google.com 网站，你可以简单地写:

```markdown
[Google][]
And then define the link:

[Google]: http://google.com/
```

[Google][]
And then define the link:

[Google]: http://google.com/

### URLs

Typora允许你以链接的形式插入url，用' < `brackets` > '包围。

`<i@typora.io>` becomes <i@typora.io>.

Typora 也会自动链接标准 url。例如: www.google.com。

### Images

图片的语法与链接类似，但它们在链接开始之前需要额外的 `!` 符号。插入图像的语法是这样的:

```markdown
![Alt text](/path/to/img.jpg)

![Alt text](/path/to/img.jpg "Optional title")
```

您可以使用拖放从图像文件或 web 浏览器插入图像。您可以通过单击图像来修改 markdown 源代码。如果使用拖放方式添加的图像与当前正在编辑的文档位于同一目录或子目录，则使用相对路径。

如果你正在使用 markdown 来构建网站，你可以在 YAML Front Matters 中 使用属性`typora-root-url`为你的本地计算机上的图像预览指定一个URL前缀。例如，在YAML Front Matters 输入`typora-root-url:/User/Abner/Website/typora.io/`，然后`![alt](/blog/img/test.png)`在 Typora 将被视为`![alt](file:///User/Abner/Website/typora.io/blog/img/test.png)`

You can find more details [here](https://support.typora.io/Images/).

### Emphasis

Markdown 将星号 (`*`) 和下划线(` _`)作为强调的指示符。用一个 `*` or `_` 包装的文本将用一个 HTML `<em>` tag 包装。例如:

```markdown
*single asterisks*

_single underscores_
```

output:

*single asterisks*

_single underscores_

GFM 会忽略单词中常用的下划线，比如:

> wow_great_stuff
>
> do_this_and_do_that_and_another_thing.

要在原本用作强调分隔符的位置产生星号或下划线，可以使用反斜杠转义:

```markdown
\*this text is surrounded by literal asterisks\*
```

推荐使用`*`符号。

### Strong

双引号`*`或`_`将导致其包含的内容被一个HTML ' <strong> '标签包装，例如

```markdown
**double asterisks**

__double underscores__
```

output:

**double asterisks**

__double underscores__

推荐使用 `**`符号

### Code

若要指示代码的内联跨度，请用反勾引号(`)将其包装起来。与预格式化的代码块不同，代码跨度表示正常段落中的代码。例如:

```markdown
Use the `printf()` function.
```

will produce:

Use the `printf()` function.

使用`printf()`函数

### Strikethrough

GFM 添加了创建划线文本的语法，这是标准 Markdown 所缺少的。

`~~Mistaken text.~~` becomes ~~Mistaken text.~~

### Underlines

Underline 是由原始 HTML 驱动的。

`<u>Underline</u>` becomes <u>Underline</u>.

### Emoji :smile:

输入带有语法的表情符号: `:smile:`。

用户可以按下`ESC` 键自动完成对表情符号的建议，或者在首选面板上启用后自动触发。此外，直接输入 UTF-8 表情符号也可以通过菜单栏中的 `Edit` -> `Emoji & Symbols`来实现。

### Inline Math

要使用此功能，请先在 `Preference` 面板-> `Markdown` 选项卡中启用它。然后，使用 `$` 包装 TeX 命令。例如: `$\lim_{x \to \infty} \exp(-x) = 0$` 将被呈现为 LaTeX 命令。

要触发内联数学的内联预览:输入“$”，然后按“ESC”键，然后输入 TeX 命令。

You can find more details [here](https://support.typora.io/Math/).

### Subscript

要使用此功能，请先在`Preference` 面板->`Markdown` 选项卡中启用它。然后，使用 `~` 包装下标内容。例如: `H~2~O`, `X~long\ text~`/

### Superscript

要使用此功能，请先在`Preference` 面板->`Markdown` 选项卡中启用它。然后，使用  `^`  包装上标内容。例如: `X^2^`.

### Highlight

要使用此功能，请先在`Preference` 面板->`Markdown` 选项卡中启用它。然后，使用 `==` 包装高亮内容。例如: `==highlight==`.

## HTML

您可以使用 HTML 来样式化纯 Markdown 不支持的内容。例如，使用 `<span style="color:red">this text is red</span>` 来添加红色文本。

### Embed Contents

一些网站提供基于框架的嵌入代码，你也可以将其粘贴到 Typora 中。例如:

```markdown
<iframe height='265' scrolling='no' title='Fancy Animated SVG Menu' src='http://codepen.io/jeangontijo/embed/OxVywj/?height=265&theme-id=0&default-tab=css,result&embed-version=2' frameborder='no' allowtransparency='true' allowfullscreen='true' style='width: 100%;'></iframe>
```

### Video

你可以使用`<video>`HTML标签来嵌入视频。例如:

```markdown
<video src="xxx.mp4" />
```

### Other HTML Support

You can find more details [here](https://support.typora.io/HTML/).

[GFM]: https://help.github.com/articles/github-flavored-markdown/

