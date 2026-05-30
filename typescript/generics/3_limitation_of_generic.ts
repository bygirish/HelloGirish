console.log("============================================================");
console.log("Limitations of Generic Code: Interfaces and Constrained Types");
console.log("============================================================");

/**
 * 
 * TypeScript will ensure that we are only able to reference properties and functions on a type of T, 
 * where these properties and functions are common across all types that are allowed for T.
 * 
 */

// Define an interface IPrintId with id property of type
// number and print method with no return value.

interface IPrintId {
  id: number;
  print(): void;
}

// Define an interface IPrintName with name property of type
// string and print method with no return value.

interface IPrintName {
  name: string;
  print(): void;
}

console.log("============================================================");
console.log("Using a generic function with interface");
console.log("============================================================");

// Define a function called "useT" that takes an argument "item" of type "T"

// function useT<T extends IPrintId | IPrintName>(item: T): void {
//   item.print();
//   item.id = 1; //error : id is not common
//   item.name = "test"; //error : name is not common
// }



