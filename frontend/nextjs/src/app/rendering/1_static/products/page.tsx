// app/products/page.tsx
import { FC } from 'react';
// import { generateStaticParams } from 'next';

interface Product {
  id: string;
  name: string;
  price: number;
}

const products: Product[] = [
  { id: '1', name: 'Product A', price: 100 },
  { id: '2', name: 'Product B', price: 200 },
];

// export async function generateStaticParams() {
//   // Generate static parameters for all product pages
//   return products.map(product => ({
//     id: product.id,
//   }));
// }

const ProductsPage: FC = () => {
  return (
    <div>
      <h1>Products</h1>
      <ul>
        {products.map(product => (
          <li key={product.id}>
            {product.name} - ${product.price}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ProductsPage;
