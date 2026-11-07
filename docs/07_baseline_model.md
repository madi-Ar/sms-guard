# Первая базовая модель SMS

Обучена LogisticRegression с class_weight='balanced', max_iter=1000, random_state=42. В единый sklearn Pipeline включены существующий create_feature_pipeline() и классификатор. Нормализация выполняется внутри извлечения признаков один раз за вызов предсказания.

Использовано сохранённое разбиение задания № 6. Train содержит 3594 SMS (3161 ham, 433 spam), validation — 770 (678 ham, 92 spam). Test не преобразуется и не оценивается. Параметры и порог не подбирались.

## Результаты validation

| Метрика | Значение |
|---|---:|
| Accuracy | 0.980519 |
| Precision spam | 0.905263 |
| Recall spam | 0.934783 |
| F1 spam | 0.919786 |

| Правильный класс | Прогноз ham | Прогноз spam |
|---|---:|---:|
| ham | TN = 669 | FP = 9 |
| spam | FN = 6 | TP = 86 |

Всего ошибок — 15. Ниже первые пять ошибок каждого типа в порядке validation. Возможные причины являются гипотезами по тексту.

## Примеры ошибок

### FN · id 2004

> TheMob>Yo yo yo-Here comes a new selection of hot downloads for our members to get for FREE! Just click & open the next link sent to ur fone...

Правильная метка 1, прогноз 0, вероятность spam 0.464111.

Реклама загрузок похожа на разговорное сообщение. Ссылка упомянута словами, явного URL в тексте нет.

### FN · id 4145

> In The Simpsons Movie released in July 2007 name the band that died at the start of the film? A-Green Day, B-Blue Day, C-Red Day. (Send A, B or C)

Правильная метка 1, прогноз 0, вероятность spam 0.099062.

Викторина про фильм сформулирована как обычный вопрос, без явной денежной суммы или URL.

### FN · id 3531

> Xmas & New Years Eve tickets are now on sale from the club, during the day from 10am till 8pm, and on Thurs, Fri & Sat night this week. They're selling fast!

Правильная метка 1, прогноз 0, вероятность spam 0.464091.

Анонс продажи билетов похож на легитимное уведомление клуба.

### FN · id 3420

> LIFE has never been this much fun and great until you came in. You made it truly special for me. I won't forget you! enjoy @ one gbp/sms

Правильная метка 1, прогноз 0, вероятность spam 0.263818.

Личная эмоциональная формулировка занимает почти весь текст; платная услуга указана только в конце.

### FN · id 4017

> You will be receiving this week's Triple Echo ringtone shortly. Enjoy it!

Правильная метка 1, прогноз 0, вероятность spam 0.211170.

Краткое уведомление о рингтоне похоже на сервисное сообщение, без явного призыва оплатить.

### FP · id 5487

> , ,  and  picking them up from various points | going 2 yeovil | and they will do the motor project 4 3 hours | and then u take them home. || 12 2 5.30 max. || Very easy

Правильная метка 0, прогноз 1, вероятность spam 0.576881.

Числа и необычный формат с разделителями могут напоминать рекламный шаблон.

### FP · id 3045

> Your bill at 3 is £33.65 so thats not bad!

Правильная метка 0, прогноз 1, вероятность spam 0.677034.

Обычное обсуждение счёта содержит денежную сумму и восклицание.

### FP · id 3408

> "HEY DAS COOL... IKNOW ALL 2 WELLDA PERIL OF STUDENTFINANCIAL CRISIS!SPK 2 U L8R."

Правильная метка 0, прогноз 1, вероятность spam 0.539267.

Заглавные буквы, числовые сокращения и восклицание похожи на признаки давления.

### FP · id 4400

> Can you tell Shola to please go to college of medicine and visit the academic department, tell the academic secretary what the current situation is and ask if she can transfer there. She should ask someone to check Sagamu for the same thing and lautech. Its vital she completes her medical education in Nigeria. Its less expensive much less expensive. Unless she will be getting citizen rates in new zealand.

Правильная метка 0, прогноз 1, вероятность spam 0.731899.

Длинный текст о переводе в другой вуз содержит слово transfer, попадающее в денежную группу словаря.

### FP · id 3228

> Rose for red,red for blood,blood for heart,heart for u. But u for me.... Send tis to all ur friends.. Including me.. If u like me.. If u get back, 1-u r poor in relation! 2-u need some 1 to support 3-u r frnd 2 many 4-some1 luvs u 5+- some1 is praying god to marry u.:-) try it....

Правильная метка 0, прогноз 1, вероятность spam 0.762890.

Цепочное личное сообщение содержит призыв переслать его, числа и восклицания.

## Запуск и использование

Из корня проекта с активированным окружением выполните:

    python -m ml.baseline
    python -m pytest -q

Эксперимент с графиком и разбором ошибок находится в notebooks/02_baseline_sms.ipynb. Выполните все ячейки по порядку, выбрав ядро виртуального окружения проекта. Если локальные данные отсутствуют, восстановите их по инструкции docs/06_feature_extraction.md.

Полный Pipeline сохраняется в ml/artifacts/sms_baseline.joblib. Пример использования после обучения:

    from ml.baseline import predict_sms

    result = predict_sms("Please bring the blue notebook tomorrow.")
    print(result["label"], result["probability"])

Функция загружает локальную модель. probability всегда означает вероятность spam (1), включая случаи, когда label равен 0. Передавать следует исходный текст, поскольку нормализация включена в Pipeline.

Предсказания и вероятности до и после сохранения полностью совпали на validation. Проверены диапазон вероятностей и сумма вероятностей классов. Новые демонстрационные SMS отсутствуют в train. Словари и IDF формируются при fit только на train. CSV и файл разбиения после эксперимента не изменились.

## Ограничения

Spam в исходном UCI-корпусе не всегда означает мошенничество. Метрики характеризуют различение ham и spam на английском языке и не доказывают качество обнаружения именно мошенничества. Русские и казахские SMS здесь не оцениваются.
