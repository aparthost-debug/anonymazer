# -*- coding: utf-8 -*-
"""
Изолированный режим: после запуска исходный массив и секретная связь строк из результата удаляются
(Приказ РКН № 140, Прил. 1, п. 1(6): исключить совместное хранение исходных и обезличенных данных).

Честное ограничение: Python/Streamlit не гарантируют затирание памяти процесса. Режим убирает ссылки и
кэши приложения; для гарантий разделяйте контуры (отдельный процесс/хост, без записи исходника на диск).
"""
import pandas as pd

ISOLATED, SHARED = "isolated", "shared (dev)"


def storage_mode(isolated):
    return ISOLATED if isolated else SHARED


def purge_source(res):
    """Удаляет из результата исходную таблицу и pair_pos. Агрегаты (метрики, гистограммы) остаются."""
    res["df_raw"] = None
    res["pair_pos"] = None
    res["purged"] = True
    return res


def result_holds_source(res, source_df):
    """Проверка для тестов/аудита: результат не ссылается на исходную таблицу и не хранит pair_pos."""
    if res.get("pair_pos") is not None:
        return True
    for v in res.values():
        if isinstance(v, pd.DataFrame) and v is source_df:
            return True
    return False
