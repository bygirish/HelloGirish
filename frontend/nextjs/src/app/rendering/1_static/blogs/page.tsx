// app/blogs/page.tsx
import { FC } from 'react';
import { BlogPost } from './types';
import fs from 'fs';
import path from 'path';

type Props = {
  posts: BlogPost[];
};

const BlogPage: FC<Props> = (props) =>  {

    console.log("posts", props);
    let posts: any[] = [];

    if (!posts) return null;

    return (
        <div>
          <h1>Blog Posts</h1>
          <ul>
            {posts.map((post) => (
              <li key={post.id}>
                <h2>{post.title}</h2>
                <p>{post.excerpt}</p>
              </li>
            ))}
          </ul>
        </div>
      );
}

export const generateStaticParams = async () => {
  const filePath = path.join(process.cwd(), 'data', 'posts.json');
  const jsonData = fs.readFileSync(filePath, 'utf-8');
  const posts: BlogPost[] = JSON.parse(jsonData);
  console.log(posts)

  return posts.map((post) => ({
    params: { id: post.id.toString() },
    props: { posts },
  }));
};

export default BlogPage;
