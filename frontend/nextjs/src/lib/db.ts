


const storage = window.localStorage;


type DBData = {
    key: string;
    value: string;
}
const saveData = (data: DBData) => {
    storage.setItem(data.key, JSON.stringify(data.value));
}


const saveMultipleData = (data: DBData[]) => {
    data.forEach((element) => {
        saveData(element);
    })
    storage.setItem
}

const getData = (key: string): string | undefined => {
    return storage.getItem(key) || undefined;
}

