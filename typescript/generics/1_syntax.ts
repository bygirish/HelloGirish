
console.log("============================================================");
console.log("Using a function with generic syntax")
console.log("============================================================");

// Define a function named "printGeneric" that takes in a generic type "T" as a parameter
function printGeneric<T>(value: T) {
  // Log the type of "T" and the value of "value" to the console
  console.log(`typeof T is : ${typeof value}`);
  console.log(`value is : ${value}`)
}

// Call the printGeneric function with various argument types

printGeneric(1);
printGeneric("test");
printGeneric(true);
printGeneric(() => { });
printGeneric({ id: 1 });


console.log("============================================================");
console.log("Explicitly specifying the type in generic syntax")
console.log("============================================================");
// Explicitly specifying the type in generic syntax
printGeneric<string>("test");

// printGeneric<number>("100")  // error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.



console.log("============================================================");
console.log("Multiple generic types")
console.log("============================================================");

// Define a function using multiple generic types
function usingTwoTypes<A, B> ( first: A, second: B) {
}

// Call the usingTwoTypes function with various argument types
usingTwoTypes<number, string> ( 1, "test");
usingTwoTypes(1, "test");
usingTwoTypes<boolean, boolean>(true, false);
usingTwoTypes("first", "second");


