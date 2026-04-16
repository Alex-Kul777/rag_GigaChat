#!/bin/bash
# Скрипт для копирования ролевого промпта в буфер обмена

if command -v clip.exe &> /dev/null; then
    # WSL/Windows
    cat .claude/roles-prompt.md | clip.exe
    echo "✅ Ролевой промпт скопирован в буфер обмена (Windows)"
elif command -v xclip &> /dev/null; then
    # Linux с xclip
    cat .claude/roles-prompt.md | xclip -selection clipboard
    echo "✅ Ролевой промпт скопирован в буфер обмена (Linux)"
elif command -v pbcopy &> /dev/null; then
    # macOS
    cat .claude/roles-prompt.md | pbcopy
    echo "✅ Ролевой промпт скопирован в буфер обмена (macOS)"
else
    echo "⚠️ Не найдена команда для копирования в буфер обмена"
    echo "Вы можете скопировать содержимое файла .claude/roles-prompt.md вручную"
fi
