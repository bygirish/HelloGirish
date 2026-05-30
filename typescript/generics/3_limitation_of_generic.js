console.log("============================================================");
console.log("Limitations of Generic Code: Interfaces and Constrained Types");
console.log("============================================================");
console.log("============================================================");
console.log("Using a generic function with interface");
console.log("============================================================");
// Define a function called "useT" that takes an argument "item" of type "T"
// function useT<T extends IPrintId | IPrintName>(item: T): void {
//   item.print();
//   item.id = 1; //error : id is not common
//   item.name = "test"; //error : name is not common
// }
