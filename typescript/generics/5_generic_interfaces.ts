console.log("============================================================");
console.log("Generic interfaces");
console.log("============================================================");

interface IPrint {
  print(): void;
}

interface ILogInterface<T extends IPrint> {
  logToConsole(iPrintObj: T): void;
}

class LogClass<T extends IPrint> implements ILogInterface<T> {
  logToConsole(iPrintObj: T): void {
    iPrintObj.print();
  }
}


// Usage example

let printObject: IPrint = {
  print() {
    console.log(`printObject.print() called`);
  },
};

let logClass = new LogClass();
logClass.logToConsole(printObject);




console.log("============================================================");
console.log("Creating new objects within generics");
console.log("============================================================");

class ClassA { }
class ClassB { }


/**
 * 
 * We can see that the compiler will not allow us to construct a new instance of the type T in this way. 
 * This is because the type of T is really of type unknown to the function at this stage.

    According to the TypeScript documentation, in order for a generic class to be able to construct an object of type T, 
    we need to refer to type T by its constructor function.
 * 
 * 
 */


// function createClassInstance<T>
//  (arg1: T): T {


// /**
//  * 
//  *  error TS2351: This expression is not constructable.
//     Type '{}' has no construct signatures.
//  * 
//  */

// //  return new arg1(); // error : see below

// }


let classAInstance = createClassInstance(ClassA);


/**
 * 
 * we have modified the arg1 parameter and are constructing an anonymous type that defines a new function 
 * and returns the type T—that is, arg1: { new() : T }. In other words, 
 * the arg parameter is a type that overloads the new function and returns an instance of T.
 */

function createClassInstance<T>
 (arg1: { new(): T }): T {
 return new arg1();
}