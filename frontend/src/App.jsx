import React from 'react';

const ResearchPage = () => {
  return (
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <div className="bg-gray-900 text-white py-16">
        <div className="max-w-6xl mx-auto px-4">
          <h1 className="text-4xl font-bold mb-4">
            LLM Reliability Framework
          </h1>
          <div className="flex gap-4 mb-8">
            <div className="text-lg">
              Authors Here
            </div>
          </div>
          <div className="flex gap-4">
            <a href="#paper" className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md">
              Paper
            </a>
            <a href="https://github.com/mmjerge/llm_reliability_framework" className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded-md">
              GitHub
            </a>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-12">
        {/* Abstract Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6">Abstract</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="prose lg:prose-lg">
              <p>
                Abstract here...
              </p>
            </div>
            <div className="bg-gray-100 p-6 rounded-lg">
              {/* Preview image or key figure */}
              <div className="aspect-video bg-gray-200 mb-4 rounded-lg">
                {/* Add your preview image here */}
              </div>
              <p className="text-sm text-gray-600">
                Figure 1: Framework Overview
              </p>
            </div>
          </div>
        </section>

        {/* Key Features Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6">Key Features</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">Feature 1</h3>
              <p className="text-gray-600">Description of feature 1...</p>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">Feature 2</h3>
              <p className="text-gray-600">Description of feature 2...</p>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-4">Feature 3</h3>
              <p className="text-gray-600">Description of feature 3...</p>
            </div>
          </div>
        </section>

        {/* Results Preview Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6">Results</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gray-50 p-6 rounded-lg">
              {/* Add your results visualization here */}
              <div className="aspect-video bg-gray-200 rounded-lg mb-4"></div>
              <h3 className="text-xl font-semibold mb-2">Key Finding 1</h3>
              <p className="text-gray-600">Description of key finding 1...</p>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              {/* Add your results visualization here */}
              <div className="aspect-video bg-gray-200 rounded-lg mb-4"></div>
              <h3 className="text-xl font-semibold mb-2">Key Finding 2</h3>
              <p className="text-gray-600">Description of key finding 2...</p>
            </div>
          </div>
        </section>

        {/* Citation Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-6">Citation</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <pre className="whitespace-pre-wrap font-mono text-sm">
              {`@article{paper,
  title={LLM Reliability Framework},
  author={Author Names},
  journal={Conference/Journal Name},
  year={2024}
}`}
            </pre>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="max-w-6xl mx-auto px-4">
          <p>© 2024 Institution. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default ResearchPage;
