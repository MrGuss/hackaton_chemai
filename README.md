# hackaton_chemai

[Ссылка на colab](https://colab.research.google.com/drive/1I7aL1cuGaNsp3vcas8X_i4SSe49UUHE3?usp=sharing) с возможностью редактирования.

## Работа с репозиторием

Клонируйте репозиторий:
```bash
git clone https://github.com/MrGuss/hackaton_chemai.git
```

Работа с репозиторием ведется через ветки. Создайте свою ветку:

```bash
git checkout -b my_branch
```

После внесения изменений сохраните и закоммитьте изменения:
```bash
git add .
git commit -m "commit message"
```

Загрузите изменения:
```bash
git push origin my_branch
```

После этого зайдите в репозиторий на github и создайте pull request.

> Перед вливанием pr обязательно скиньте его в чат в телеграме.

## Работа с кодом

Создайте venv:
```bash
python -m pip install venv
python -m venv .venv
```

Активируйте venv:

> Linux/mac

```bash
source .venv/bin/activate
```
> Windows powershell
```powershell
.\venv\Scripts\Activate.ps1
```
> Windows cmd
```cmd
.\venv\Scripts\activate
```

Установите зависимости:
```bash
pip install -r requirements.txt
```

Далее используйте любой удобный редактор jupyter notebook.

## Troubleshooting

Если при запуске `git push` вы получете ошибку `error: failed to push some refs to 'github.com:MrGuss/hackaton_chemai.git`, то нужно влить изменения из удаленного репозитория в вашу локальную ветку. Для этого выполните следующее:

```bash
git config pull.rebase false # Эту команду нужно выполнить только один раз для настройки локального репозитория
git pull # Эта команда скачивает изменения из удаленного репозитория
```

После этого выполните команду `git push` заново.
Использовать `git push --force` не рекомендуется.